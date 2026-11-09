"""
AI engine configuration (Settings page).

Provider-agnostic controls for the AI providers (local Ollama by default; Google Gemini, OpenAI, Mistral
and Anthropic Claude as cloud providers that require an explicit data-sharing consent; Claude has no
embedding model, so another provider embeds the documents), model and embedding
selection, the document library, retrieval parameters and engine initialisation. Laid out as cards:
provider and models, data sharing and document library on the left; retrieval, the vector database
(LanceDB or Qdrant, see ``manage.vectorstores``) and the engine on the right. Widget keys are stable
because other pages read them from the session.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from core.registry import Selection, as_code
from core.workspace import remove_pdf, safe_filename
from manage.ai_payload import (
    cloud_model,
    host_of,
    on_this_machine,
    outside_private_network,
    provider_label,
    recipient,
    server_recipient,
    store_recipient,
)
from manage.iam.authz import is_admin_role
from manage.rag import ConnectorFactory, ProviderConfig, ProviderType
from manage.rag.providers import (
    AI_PROVIDERS,
    declared,
    default_provider,
    embedding_providers,
    file_readers,
    provider_setting,
    server_key,
)
from manage.server_addresses import (
    ADDRESS_REMOVED,
    NAME_MAX_LENGTH,
    OLLAMA_ADDRESSES,
    QDRANT_ADDRESSES,
    Added,
    AddressList,
    NotAllowed,
    address_name,
)
from manage.vectorstores import (
    VECTOR_STORES,
    IndexManifest,
    MachineProfile,
    VectorEngine,
    VectorStoreSettings,
    estimate_chunks,
    probe_machine,
    qdrant_embedded_warnings,
    recommend,
)
from manage.vectorstores.qdrant_index import (
    QdrantIndex,
    adopt_earlier_index,
    explicit_address,
    server_folder,
    server_indexes,
    unrecorded_collection,
)
from ui import preferences
from ui.components import confirm
from ui.components.connector_selector import connector_selector, explain_choice
from ui.components.text import plain
from ui.services import ai_engine
from ui.session_scope import get_current_scope
from utils.data_paths import get_vectors_dir
from utils.settings import ADDRESS_MAX_LENGTH

# .env is loaded once, at start-up (main.py): the keys are in the environment from the first page on
# The choices are kept for the whole session: Streamlit drops the value of a widget whose page is not shown, so a
# revisit showed Ollama, the first model and an empty consent box, and withdrew the running engine's consent
PERSIST = "session"
STOP_NOTICE = (
    "Confirmation withdrawn: the AI engine stopped. Nothing more is sent from this tab; what was already sent stays "
    "with the provider, under its own terms."
)
# The providers, their names, keys and models are declared once (manage.rag.providers.AI_PROVIDERS): this card reads
# them, in the registry's order


def _documents_dir() -> str:
    """The signed-in user's document library (audit E4); the legacy shared folder outside a session."""
    scope = get_current_scope()
    return str(scope.documents_dir()) if scope else "DOCUMENTS"


@st.cache_data(ttl=60, show_spinner=False)
def _discover_ollama_models(base_url: str) -> list[str]:
    """Ollama model list, probed at most once a minute per endpoint (audit E11)."""
    try:
        cfg = ProviderConfig(provider=ProviderType.OLLAMA, base_url=base_url)
        return list(ConnectorFactory.get_llm_connector(cfg).get_available_models() or [])
    except Exception:
        return []


@st.cache_data(ttl=60, show_spinner=False)
def _runs_at(base_url: str, model: str) -> str | None:
    """Where the Ollama server runs a model when not itself (a cloud model, a copy of one), asked at most once a
    minute per endpoint and model, as the engine asks it when it starts. A silence raises NoAnswer, which is never
    kept (a call that raises is not cached): Settings never states where a model runs from a server that did not say."""
    from manage.rag.providers.ollama_server import runs_at

    return runs_at(base_url, model)


@st.cache_data(ttl=60, show_spinner=False)
def _remote_models(base_url: str) -> dict[str, str]:
    """The models the Ollama server lists that it runs elsewhere, to mark them in the list."""
    from manage.rag.providers.ollama_server import remote_models

    return remote_models(base_url)


@st.cache_data(ttl=3600, show_spinner=False)
def _discover_cloud_models(_manager, provider: str, api_key: str) -> list[str]:
    """Cloud model list, fetched at most once an hour per key (audit E11)."""
    try:
        return list(_manager.get_available_models(provider) or [])
    except Exception:
        return []


def _api_key_input(provider: str, label: str, key: str, help_text: str) -> str | None:
    """Session-scoped API key entry.

    The server-side key is never shown or pre-filled and nothing is written to ``os.environ``
    (audit E6). Leaving the field empty falls back to the server-configured key, if any.
    """
    typed = st.text_input(
        label, value="", type="password", help=help_text, key=key, icon=":material/key:", persist_state=PERSIST
    )
    if server_key(provider):
        st.caption("A server-side key is configured; leave this empty to use it.")
    st.session_state.rag_manager.api_key = typed or None
    return typed or None


def _ensure_manager() -> None:
    ai_engine()  # built on first use, and again after a code change (ui.services)


def _running() -> dict:
    """The provider and chat model of the engine that runs (empty when it does not run)."""
    manager = st.session_state.rag_manager
    if not getattr(manager, "initialized", False):
        return {}
    return {
        "provider": getattr(manager, "provider", None),
        "model": getattr(getattr(getattr(manager, "llm", None), "config", None), "model_name", None),
    }


def _address_badge(url: str, carried: str = "the prompts and documents") -> None:
    """Where an address (an Ollama's, a Qdrant server's) sends the data: the kind of place in a badge, the host in a
    caption that wraps (in a badge, a Qdrant Cloud host was cut before its region, which says where the data go)."""
    if on_this_machine(url):
        st.badge("Local: no data leaves this server", icon=":material/lock:", color="green")
        return
    if outside_private_network(url):
        st.badge("Outside your network", icon=":material/public:", color="red")
    else:
        st.badge("Your network", icon=":material/lan:", color="orange")
    text = url.strip()
    scheme = urlparse(text).scheme.lower() if "//" in text else ""  # none given: the client uses http
    said = f"Data goes to {_shown(host_of(url) or text)}."
    if scheme in ("", "http"):
        said += f" This address uses http: {carried} cross the network unencrypted."
    st.caption(said)


def _is_admin() -> bool:
    return is_admin_role(st.session_state.get("user_role"))


def _account() -> str:
    """The signed-in account's name, as the user signs in (the workspace folder may be a code of it)."""
    return str(st.session_state.get("username") or "")


def _forget(name: str) -> None:
    st.session_state.pop(name, None)


# the preference that keeps each address choice for the next session (manage.preferences)
_PREFERENCES = {
    "rag_ollama_base_url": "ollama_address",
    "rag_claude_ollama_base_url": "ollama_embedding_address",
    "rag_qdrant_url": "qdrant_server",
}


def _remember(key: str, place: "_Place") -> None:
    """Keep the address chosen in ``key`` for the next session: only one the list holds ("" for the embedded Qdrant).
    An on_change runs before the page, while the session holds the browser's raw value: a forged one is never kept."""
    value = st.session_state.get(key)
    if isinstance(value, str) and value in _options(place):
        preferences.save(**{_PREFERENCES[key]: value})


def _chosen(key: str, place: "_Place") -> None:
    """A choice's on_change: the note of a removed choice goes, and the new one is kept for the next session."""
    _forget(f"{key}_dropped")
    _remember(key, place)


def _shown(address: str) -> str:
    return as_code(address, limit=ADDRESS_MAX_LENGTH)


@dataclass(frozen=True)
class _Place:
    """What Settings needs to offer one kind of server's addresses, its sentences written in full: its list, the
    settings that set the deployment's (the first address's, then the others'), the option that uses no server (Qdrant's
    embedded mode), and what is said."""

    addresses: AddressList
    variables: tuple[str, ...]
    list_name: str  # "Ollama addresses": the button "Manage Ollama addresses", and the messages
    chosen_noun: str  # "The Ollama address you chose, ..., is no longer offered"
    server_noun: str  # "Your Ollama server is not listed?"
    caption: str  # at the top of the administrators' list
    example: str
    name_example: str
    no_server: str | None = None
    empty: str = ""  # the administrators' list when nothing is listed
    left_behind: str = ""  # what stays on a server once removed, said at its removal


OLLAMA_PLACE = _Place(
    OLLAMA_ADDRESSES,
    ("OLLAMA_BASE_URL", "CMP_OLLAMA_ADDRESSES"),
    list_name="Ollama addresses",
    chosen_noun="Ollama address",
    server_noun="Ollama server",
    caption="Users choose the Ollama address from this list; only an administrator adds one. Add only an Ollama service "
    "you trust with the prompts and documents sent to it.",
    example="http://ollama.example.org:11434",
    name_example="GPU server, Biostatistics",
)
QDRANT_PLACE = _Place(
    QDRANT_ADDRESSES,
    ("CMP_QDRANT_URL",),
    list_name="Qdrant servers",
    chosen_noun="Qdrant server",
    server_noun="Qdrant server",
    caption="Users choose a Qdrant server from this list, or the embedded Qdrant; only an administrator adds one. Add "
    "only a Qdrant server you trust with the passages of the PDFs indexed there, with their names, and the vector of "
    "each search. A server that asks for an API key: each user types it under **Qdrant API key**, in each session, "
    "and no key is kept here; a shared server with a key belongs in the deployment's settings (`CMP_QDRANT_URL`, "
    "`CMP_QDRANT_API_KEY`), where its key stays on this server.",
    example="https://qdrant.example.org:6333",
    name_example="Qdrant Cloud, Frankfurt",
    no_server="Embedded in your workspace (no server)",
    empty="No Qdrant server is listed yet: users have the embedded Qdrant only.",
    left_behind=" The indexes users built there stay on that server: each user's card names the collection left "
    "there, for that server's administrator to delete.",
)


def _options(place: _Place) -> list[str]:
    return ([""] if place.no_server else []) + place.addresses.listed()


def _named(value: str, names: dict[str, str] | None = None) -> str:
    """A choice in a sentence: its name first, when an administrator gave one, then the address as code; or the
    server-free option."""
    if not value:
        return "the embedded Qdrant, in your workspace"
    name = (names or {}).get(value)
    return f"{plain(name, limit=NAME_MAX_LENGTH)} ({_shown(value)})" if name else _shown(value)


def _labelled(value: str, names: dict[str, str]) -> str:
    """An option of the list: its name, when an administrator gave one, then the address (cut in its middle)."""
    return f"{names[value]} · {_short(value)}" if value in names else _short(value)


def _seed_choice(key: str, place: _Place) -> None:
    """
    A choice's first value in a session: the one kept from the last session while the list holds it. A kept one the
    list no longer holds is named, and the first option is kept in its place, as a removal within a session does (for
    Qdrant, the embedded one, never the deployment's server). Else the deployment's first address (Qdrant: its server,
    else the embedded Qdrant).
    """
    if key in st.session_state:
        return
    options = _options(place)
    saved = getattr(preferences.current(), _PREFERENCES[key], None)
    if saved is not None and saved in options:
        st.session_state[key] = saved
        return
    if saved:  # kept, then removed: named once, the first option kept in its place
        st.session_state[f"{key}_dropped"] = (str(saved), True)
        st.session_state[key] = options[0]
        _remember(key, place)
        return
    st.session_state[key] = (place.addresses.deployment() or options)[0]


def _short(address: str, limit: int = 44) -> str:
    """An address in a narrow list, cut in its middle past ``limit`` characters: its end (a Qdrant Cloud host's region
    and port) stays readable."""
    if len(address) <= limit:
        return address
    keep = (limit - 1) // 2
    return f"{address[:keep]}…{address[-keep:]}"


def _address_choice(key: str, label: str, help_text: str, place: _Place) -> str:
    """
    A server's address chosen from the server's list (manage.server_addresses), never typed: the server sends no
    request to an address a user writes, which on a shared server could reach a machine of its network. A value the
    list does not hold is never used: an address chosen, then removed by an administrator, is named with the one used
    instead; one the browser forged is replaced by the first option. The caller draws the administrators' list
    (``_manage_addresses``) below the choice and its badge.
    """
    _seed_choice(key, place)
    options = _options(place)
    names = place.addresses.names()
    prior = st.session_state.get(key)
    if prior is not None and prior not in options:  # removed since it was chosen: named until another is chosen
        st.session_state[f"{key}_dropped"] = (str(prior), False)
    chosen = st.selectbox(
        label,
        options,
        index=0,
        key=key,
        format_func=lambda value: _labelled(value, names) if value else place.no_server or "",
        help=help_text
        + (
            f" Add another under Manage {place.list_name}, below."
            if _is_admin()
            else " Only a CohortStudio administrator adds one."
        ),
        filter_mode=None,  # a list to choose from: nothing is typed here
        on_change=_chosen,
        args=(key, place),
        persist_state=PERSIST,
    )
    if chosen not in options:  # the widget's value comes from the browser: only an option offered goes on
        chosen = options[0]
    dropped = st.session_state.get(f"{key}_dropped")
    if dropped:  # (the address, whether it was kept from the last session); its name went with it
        address, kept = dropped
        st.caption(
            f"The {place.chosen_noun} {'kept from your last session' if kept else 'you chose'}, {_shown(address)}, is "
            f"no longer offered: this page uses {_named(str(chosen), names)}."
        )
    if not _is_admin():
        st.caption(f"Your {place.server_noun} is not listed? Ask a CohortStudio administrator to add it.")
    return str(chosen)


def _day(moment: str) -> str:
    """A date in a row, never broken across two lines ("9 Nov 2026", its spaces no-break ones)."""
    try:
        when = datetime.fromisoformat(moment)
    except ValueError:
        return ""
    return chr(0xA0).join((str(when.day), f"{when:%b}", f"{when:%Y}"))


def _manage_addresses(key: str, place: _Place) -> None:
    """
    The administrators' list of a server's addresses every user may choose, below the choice and its badge: add one
    (offered at once, and chosen here; Enter adds), rename or remove one added here; the deployment's are changed in its
    settings.
    The form and its outcome come first, the list after, in a box of its own height, so that the popover keeps one size.
    """
    if not _is_admin():
        return
    with st.popover(f"Manage {place.list_name}", icon=":material/edit:"):
        st.caption(place.caption)
        if place.addresses.unreadable():
            st.warning(
                "The list of added addresses cannot be read (see the server log): only the deployment's are offered. "
                f"Repair or remove `server/{place.addresses.kind}_addresses.json`, in the data folder, before adding one."
            )
        with st.form(f"{key}_add_form", border=False, enter_to_submit=True):
            # the name first: the address pasted last, Enter sends both
            st.text_input(
                "Name (optional)",
                key=f"{key}_name",
                placeholder=place.name_example,
                max_chars=NAME_MAX_LENGTH,
                help="Name the service, never a person: every user sees it before the address, and the badge, not the "
                "name, says where the data go. To change it later, use Rename on its row.",
            )
            box, button = st.columns([4, 1], vertical_alignment="bottom")
            box.text_input("Address to add", key=f"{key}_new", placeholder=place.example, max_chars=ADDRESS_MAX_LENGTH)
            button.form_submit_button(
                "Add", key=f"{key}_add", icon=":material/add:", on_click=_add_address, args=(key, place)
            )
        said = st.session_state.pop(f"{key}_said", None)
        if said:
            {"ok": st.success, "info": st.info, "error": st.error}[said[0]](said[1])
        removed = st.session_state.pop(f"{key}_removed", None)
        if removed:
            st.button("Add it back", key=f"{key}_add_back", on_click=_add_back, args=(key, place, removed))
        deployed, added = place.addresses.deployment(), place.addresses.added()
        with st.container(height=180, border=False):
            if not deployed and not added and place.empty:
                st.caption(place.empty)
            for n, address in enumerate(deployed):
                variable = place.variables[0] if n == 0 else place.variables[-1]
                st.markdown(f"{_shown(address)} · set by the deployment (`{variable}`)")
            for n, item in enumerate(added):
                if st.session_state.get(f"{key}_renaming") == item.address:
                    _rename_form(key, place, item)
                    continue
                text, rename, remove = st.columns([3, 1, 1], vertical_alignment="center")
                when = _day(item.added_at)
                text.markdown(
                    (f"**{plain(item.name, limit=NAME_MAX_LENGTH)}** · " if item.name else "")
                    + f"{_shown(item.address)} · added by {plain(item.added_by, limit=64)}"
                    + (f", {when}" if when else "")
                )
                rename.button("Rename", key=f"{key}_rename_{n}", on_click=_start_rename, args=(key, item))
                remove.button(
                    "Remove", key=f"{key}_remove_{n}", on_click=_remove_address, args=(key, place, item.address)
                )


def _rename_form(key: str, place: _Place, item: Added) -> None:
    """The row of the address being renamed: a box holding its name (left empty, no name), saved with Enter."""
    with st.form(f"{key}_rename_form", border=False, enter_to_submit=True):
        st.text_input(
            f"New name for {_shown(item.address)} (empty: no name)",
            key=f"{key}_rename_name",
            placeholder=place.name_example,
            max_chars=NAME_MAX_LENGTH,
        )
        with st.container(horizontal=True):
            st.form_submit_button(
                "Save", key=f"{key}_rename_save", type="primary", on_click=_rename_address, args=(key, place, item)
            )
            st.form_submit_button("Cancel", key=f"{key}_rename_cancel", on_click=_forget, args=(f"{key}_renaming",))


def _start_rename(key: str, item: Added) -> None:
    """Rename's callback: the row turns into a box holding the address's name."""
    st.session_state[f"{key}_renaming"] = item.address
    st.session_state[f"{key}_rename_name"] = item.name


def _rename_address(key: str, place: _Place, item: Added) -> None:
    """Save's callback: the new name, for every user ("" removes it); the address this page uses is left as it is."""
    scope = get_current_scope()
    try:
        if scope is None:
            raise NotAllowed("only an administrator renames an address")
        held = place.addresses.rename(
            item.address, str(st.session_state.get(f"{key}_rename_name") or ""), by=scope, account=_account()
        )
    except (ValueError, PermissionError) as exc:  # the box keeps what was typed
        st.session_state[f"{key}_said"] = ("error", f"Not renamed: {exc}.")
        return
    _forget(f"{key}_renaming")
    names = place.addresses.names()
    if not held:
        said = ("info", f"{_shown(item.address)} is not among the {place.list_name} any more.")
    elif item.address in names:
        said = ("ok", f"Renamed in every user's list: {_named(item.address, names)}.")
    else:
        said = ("ok", f"Its name is removed from every user's list: {_shown(item.address)}.")
    st.session_state[f"{key}_said"] = said


def _choose(key: str, place: _Place, address: str) -> None:
    """Choose ``address`` on this page, and keep it for the next session, when the list holds it."""
    if address in _options(place):
        st.session_state[key] = address
        st.session_state.pop(f"{key}_dropped", None)
        _remember(key, place)


def _add_address(key: str, place: _Place) -> None:
    """
    Add's callback (Enter in either box too): the address typed is stored for every user (an administrator only), and
    chosen here. One already listed is left chosen or not: a name typed for one added here renames it, and one the
    deployment sets takes no name.
    """
    scope = get_current_scope()
    name = str(st.session_state.get(f"{key}_name") or "")
    try:
        if scope is None:
            raise NotAllowed("only an administrator adds an address")
        before, named_before = place.addresses.listed(), place.addresses.names()
        offered = place.addresses.add(
            str(st.session_state.get(f"{key}_new") or ""), by=scope, account=_account(), name=name
        )
    except (ValueError, PermissionError) as exc:  # the message never holds the address
        st.session_state[f"{key}_said"] = ("error", f"Not added: {exc}.")
        return
    st.session_state[f"{key}_new"] = ""
    st.session_state[f"{key}_name"] = ""
    names = place.addresses.names()
    if offered not in before:
        _choose(key, place, offered)
        said = ("ok", f"Added to the {place.list_name} of every user, and chosen here: {_named(offered, names)}.")
    elif names.get(offered) != named_before.get(offered):
        said = ("ok", f"Renamed in every user's list: {_named(offered, names)}.")
    elif address_name(name) and offered in place.addresses.deployment():
        said = (
            "info",
            f"Not named: {_shown(offered)} is set by the deployment, and only an address added here takes a name.",
        )
    else:
        said = ("info", f"Already among the {place.list_name}: {_named(offered, names)}.")
    st.session_state[f"{key}_said"] = said


def _add_back(key: str, place: _Place, removed: tuple[str, str, bool]) -> None:
    """Add it back's callback: the address just removed, added again with its name; chosen again only on the page that
    had it chosen, as it was."""
    address, name, was_chosen = removed
    scope = get_current_scope()
    try:
        if scope is None:
            raise NotAllowed("only an administrator adds an address")
        offered = place.addresses.add(address, by=scope, account=_account(), name=name)
    except (ValueError, PermissionError) as exc:
        st.session_state[f"{key}_said"] = ("error", f"Not added back: {exc}.")
        return
    if was_chosen:
        _choose(key, place, offered)
    shown = _named(offered, place.addresses.names())
    st.session_state[f"{key}_said"] = (
        "ok",
        f"Added back to the {place.list_name} of every user{', and chosen here again' if was_chosen else ''}: {shown}.",
    )


def _remove_address(key: str, place: _Place, address: str) -> None:
    """Remove's callback: every user's AI engine on the address stops at its next use (ui.services); this page moves to
    the first option when it had chosen this one, and says so."""
    scope = get_current_scope()
    name = place.addresses.names().get(address, "")
    try:
        removed = scope is not None and place.addresses.remove(address, by=scope, account=_account())
    except PermissionError as exc:
        st.session_state[f"{key}_said"] = ("error", f"Not removed: {exc}.")
        return
    if st.session_state.get(f"{key}_renaming") == address:
        _forget(f"{key}_renaming")
    if not removed:
        st.session_state[f"{key}_said"] = ("info", f"{_shown(address)} is not among the {place.list_name} any more.")
        return
    said = (
        f"Removed from the {place.list_name} of every user: {_named(address, {address: name})}. An AI engine started "
        "on it stops at its next use." + place.left_behind
    )
    was_chosen = st.session_state.get(key) == address
    if was_chosen:
        first = _options(place)[0]
        st.session_state[key] = first
        _remember(key, place)
        said += f" This page now uses {_named(first, place.addresses.names())}."
    st.session_state[f"{key}_said"] = ("ok", said)
    st.session_state[f"{key}_removed"] = (address, name, was_chosen)  # for Add it back


# ----------------------------------------------------------------------------- cards

def _provider_card() -> dict:
    """Provider, connection and model choice; returns the engine configuration."""
    cfg: dict = {"api_key": None, "base_url": None}
    running = _running()
    prefs = preferences.current()  # the choices kept from the last session (never a key or a consent)
    with st.container(border=True):
        st.subheader("Provider and models", icon=":material/smart_toy:", anchor=False)
        # the running engine's, then the saved one, then CMP_AI_PROVIDER (local Ollama by default): a cloud provider
        # chosen so is only preselected, and receives nothing until the session confirms it (the consent card)
        chosen = connector_selector(
            AI_PROVIDERS,
            key="rag_provider_select",
            label="AI provider",
            preference=prefs.ai_provider,
            preference_name="ai_provider",
            setting=provider_setting(),
            seed=running.get("provider"),
            offer_auto=False,
            persist=PERSIST,
            help_text="Ollama runs on this server or your network, or sends a cloud model's prompts to ollama.com; the "
            "cloud providers need an API key and your confirmation that data may be sent to them.",
        )
        provider = chosen.chosen or default_provider()  # none can run here: named all the same
        cfg["provider"] = provider
        info = declared(provider)  # its key, its models (manage.rag.providers)
        where = None  # an Ollama's: where the data goes, drawn once the model is known
        if provider == "ollama":
            cfg["base_url"] = _address_choice(
                "rag_ollama_base_url", "Ollama address", "The Ollama services this server offers.", OLLAMA_PLACE
            )
            where = st.container()  # where the data goes, once the model is known: a cloud model runs elsewhere
            _manage_addresses("rag_ollama_base_url", OLLAMA_PLACE)  # below the address's badge
            chat_options = _discover_ollama_models(cfg["base_url"]) or list(info.chat_models)
            embed_options = list(info.embedding_models)
            chat_key, embed_key = "rag_ollama_model_select", "rag_ollama_embed_select"
            chat_help = (
                "Model for question answering and taxonomy generation. A model marked (ollama.com) in the list "
                "runs on that host through your Ollama server: what it is sent needs your confirmation."
            )
            embed_help = "Local embedding model (pull it first: 'ollama pull <model>')."
        else:
            st.badge("Cloud: requires your confirmation", icon=":material/cloud:", color="orange")
            key_help = f"Required for {info.display_name} inference."
            if info.lists_models:  # _discover_cloud_models: the typed key, before any confirmation
                key_help += (
                    f" Typing a key asks {provider_label(provider)} which models it may use: the key is sent, none "
                    "of your data."
                )
            cfg["api_key"] = _api_key_input(
                provider, info.key_label or "API key", f"rag_{provider}_api_key", key_help
            )
            chat_options = list(info.chat_models)
            if info.lists_models and cfg["api_key"] and hasattr(
                st.session_state.rag_manager, "get_available_models"
            ):
                for model in _discover_cloud_models(st.session_state.rag_manager, provider, cfg["api_key"]):
                    if model not in chat_options:
                        chat_options.append(model)
            embed_options = list(info.embedding_models)
            chat_key, embed_key = f"rag_{provider}_model_select", f"rag_{provider}_embed_select"
            chat_help = "Model used for question answering and taxonomy generation."
            embed_help = "Embedding model used to index your documents."
            if provider == "anthropic":
                chat_help += " The most capable model is listed first; the app's temperature setting does not apply to Claude."
            if not info.embeds():  # it answers; another provider embeds the documents
                embed_options, embed_help = _claude_embedding_block(cfg)
                embed_key = f"rag_{provider}_{cfg['embedding_provider']}_embed_select"
        engine_model = running.get("model") if running.get("provider") == provider else None
        embed_owner = cfg.get("embedding_provider") or provider  # Claude's documents are embedded by another provider
        stale = [
            *_seed_model(chat_key, chat_options, engine_model, prefs.ai_chat_model.get(provider), "chat"),
            *_seed_model(embed_key, list(embed_options), None, prefs.ai_embedding_model.get(embed_owner), "embedding"),
        ]
        col_chat, col_embed = st.columns(2)
        elsewhere = _remote_models(cfg["base_url"] or "") if provider == "ollama" else {}

        def model_label(model: str) -> str:  # an Ollama model run elsewhere, said in the list
            return f"{model} (ollama.com)" if provider == "ollama" and (cloud_model(model) or model in elsewhere) else model

        cfg["model"] = col_chat.selectbox(
            "Chat model", chat_options, index=0, key=chat_key, help=chat_help, persist_state=PERSIST,
            format_func=model_label,
            on_change=preferences.save_from, args=(chat_key, "ai_chat_model"), kwargs={"provider": provider},
        )
        silent = False  # an Ollama server that did not say where it runs the model
        if provider == "ollama":  # as the engine asks when it starts: a copy of a cloud model reads local by name
            from manage.rag.providers.ollama_server import NoAnswer

            try:
                cfg["remote_host"] = _runs_at(cfg["base_url"] or "", str(cfg["model"] or ""))
            except NoAnswer as exc:  # the list's own mark when it has one (the box is then offered), else not known
                cfg["remote_host"] = elsewhere.get(str(cfg["model"] or ""))
                silent = cfg["remote_host"] is None
                cfg["silence"] = getattr(exc, "reason", "")
        cfg["where_unknown"] = silent  # read by the Data sharing card: never said to keep the data here
        cfg["embedding"] = col_embed.selectbox(
            "Embedding model", embed_options, index=0, key=embed_key, help=embed_help, persist_state=PERSIST,
            on_change=preferences.save_from, args=(embed_key, "ai_embedding_model"), kwargs={"provider": embed_owner},
        )
        for notice in stale:
            st.caption(notice)
        if where is not None:
            with where:
                if cloud_model(cfg["model"]) or cloud_model(cfg["embedding"]) or cfg.get("remote_host"):
                    # pulled from ollama.com, or a copy of such a model, it runs there: the address is only the way
                    host = host_of(cfg.get("remote_host")) or "ollama.com"
                    st.badge(f"Cloud model: data goes to {host}", icon=":material/cloud:", color="orange")
                    st.caption(
                        f"This model runs on {host} through your Ollama server: what it is sent leaves this machine, "
                        "and needs your confirmation below."
                    )
                    if not on_this_machine(cfg["base_url"]):  # the server at the address receives it too
                        _address_badge(cfg["base_url"])
                elif silent:  # said as not known, never as "Local": the start asks again, and starts on an answer
                    st.badge("Where this model runs is not known", icon=":material/help:", color="gray")
                    st.caption(_silence(cfg) + " An engine started with it asks again, and starts only once the "
                               "server answers.")
                    if not on_this_machine(cfg["base_url"]):  # what the address says is known all the same
                        _address_badge(cfg["base_url"])
                else:
                    _address_badge(cfg["base_url"])
    return cfg


def _silence(cfg: dict) -> str:
    """Why a server's answer on where it runs the chosen model is missing, the model and the address named."""
    from manage.rag.providers.ollama_server import UNREACHABLE

    model = as_code(str(cfg.get("model") or "the model"))
    url = str(cfg.get("base_url") or "")
    address = (urlparse(url).netloc if "//" in url else url) or "localhost:11434"
    if cfg.get("silence") == UNREACHABLE:
        return f"The Ollama server did not answer at {as_code(address)}: where it runs {model} is not known."
    return f"The Ollama server did not say where it runs {model}."


def _seed_model(
    key: str, options: list[str], running: str | None, saved: str | None, kind: str
) -> list[str]:
    """
    The model a control starts from, when the session has none: the running engine's, then the one kept from the last
    session. A saved model that is no longer offered is not used: the notice returned says so, and the file keeps it
    until the user chooses again.
    """
    if key in st.session_state or not options:
        return []
    if running in options:
        st.session_state[key] = running
    elif saved in options:
        st.session_state[key] = saved
    elif saved:
        return [
            f"Your saved {kind} model {preferences.shown(saved)} is not offered here: "
            f"{preferences.shown(options[0])} is used."
        ]
    return []


def _claude_embedding_block(cfg: dict) -> tuple[list[str], str]:
    """Claude has no embedding model: choose the provider that embeds the documents (local Ollama by default)."""
    st.caption(
        "Claude answers; another provider embeds your documents for search. With Ollama your Ollama service "
        "indexes the whole library (on this machine, or at the address below), and only the passages retrieved "
        "for a question are sent to Claude; Document insight sends it the text of the document whose graph it "
        "extracts."
    )
    saved = preferences.current().ai_claude_embedding_provider
    embedders = embedding_providers()  # the providers that embed, local Ollama first
    if "rag_claude_embed_provider" not in st.session_state and saved in embedders:
        st.session_state["rag_claude_embed_provider"] = saved  # the choice kept from the last session
    elif saved and saved not in embedders:
        st.caption(f"Your saved embedding provider {preferences.shown(saved)} is not offered here.")
    embed_provider = st.selectbox(
        "Embedding provider",
        embedders,
        format_func=AI_PROVIDERS.display_name,
        index=0,
        key="rag_claude_embed_provider",
        on_change=preferences.save_from,
        args=("rag_claude_embed_provider", "ai_claude_embedding_provider"),
        help="Who turns your documents into vectors for search. Ollama keeps the documents on this machine or at "
        "its address; a cloud provider needs its own API key and receives the text of the indexed PDFs and every "
        "search text.",
        persist_state=PERSIST,
    )
    cfg["embedding_provider"] = embed_provider
    if embed_provider == "ollama":
        cfg["embedding_base_url"] = _address_choice(
            "rag_claude_ollama_base_url",
            "Ollama address (embeddings)",
            "The Ollama services this server offers, to embed the documents.",
            OLLAMA_PLACE,
        )
        _address_badge(cfg["embedding_base_url"])
        _manage_addresses("rag_claude_ollama_base_url", OLLAMA_PLACE)
        return list(declared(embed_provider).embedding_models), "Local embedding model (pull it first: 'ollama pull <model>')."
    info = declared(embed_provider)
    typed = st.text_input(
        f"{info.key_label or 'API key'} (embeddings)",
        value="",
        type="password",
        key=f"rag_claude_{embed_provider}_embed_key",
        icon=":material/key:",
        help=f"Key for {info.display_name} embeddings. Never stored; empty uses the server key if configured.",
        persist_state=PERSIST,
    )
    if server_key(embed_provider):
        st.caption("A server-side key is configured for the embeddings; leave this empty to use it.")
    cfg["embedding_api_key"] = typed or None
    return list(info.embedding_models), f"{info.display_name} embedding model used to index your documents."


def _roles(
    cfg: dict, vector: dict | None = None
) -> tuple[str | None, str | None, str | None, str | None]:
    """
    The recipients that need the confirmation, by role: the chat provider when it is a cloud one, an Ollama model run
    elsewhere (its tag, or the server's ``remote_host``) or an Ollama outside the private network; the embedding
    provider of Claude when it is another such recipient (it receives the documents and every search text); the
    Qdrant server outside the private network that keeps the index; the Ollama server outside the private network
    when its model runs elsewhere (it receives the prompts too).
    """
    chat = recipient(cfg["provider"], cfg.get("base_url"), cfg.get("model"), remote_host=cfg.get("remote_host"))
    server = server_recipient(cfg["provider"], cfg.get("base_url"))
    embedding = recipient(
        cfg.get("embedding_provider") or cfg["provider"],
        cfg.get("embedding_base_url") or cfg.get("base_url"),
        cfg.get("embedding"),
    )
    on_qdrant = bool(vector) and vector.get("vector_engine") == VectorEngine.QDRANT
    store = store_recipient(vector.get("qdrant_url")) if on_qdrant else None
    server = server if server != chat else None
    return chat, (embedding if embedding not in (chat, server) else None), store, server


def _recipients(cfg: dict, vector: dict | None = None) -> list[str]:
    """The recipients of a configuration that need the confirmation, as one box confirms them together."""
    return [r for r in _roles(cfg, vector) if r]


def _names(recipients: list[str], embedding: str | None = None) -> str:
    return " and ".join(f"{provider_label(r)}{' (embeddings)' if r == embedding else ''}" for r in recipients)


def _consent_key(recipients: list[str]) -> str:
    """The session key of the confirmation of exactly these recipients."""
    return f"rag_consent::{'+'.join(recipients)}"


def _consent_recipients(key: str) -> list[str]:
    return key.removeprefix("rag_consent::").split("+")


def _consent_changed(key: str, recipients: tuple[str, ...]) -> None:
    """
    The user ticked or unticked a box: only then does the confirmation change. Acting on how a box is shown made
    an untouched box for another configuration withdraw, and stop, the engine that runs.
    """
    manager = st.session_state.rag_manager
    if st.session_state.get(key):
        manager.confirm(*recipients)
        return
    if manager.withdraw_consent(*recipients):
        st.session_state["rag_stop_notice"] = STOP_NOTICE
    # every confirmation naming these recipients went: so do the ticks of their other boxes, which would otherwise
    # confirm them again when shown
    for other in [k for k in st.session_state if isinstance(k, str) and k.startswith("rag_consent::")]:
        if other != key and set(_consent_recipients(other)) & set(recipients):
            st.session_state[other] = False


def _notice(recipients: list[str], cfg: dict, vector: dict | None = None) -> str:
    """What the recipients of one box receive, each for its role (a Qdrant server receives no column summary)."""
    _chat, embedding, store, _server = _roles(cfg, vector)
    lines = []
    if any(r not in (embedding, store) for r in recipients):  # the provider that answers
        lines += [
            (
                "- **About your data, a summary of each column:** its name and type; the 5th, 25th, 50th, 75th and "
                "95th percentiles of numeric columns of more than 10 values, rounded to two significant digits of the "
                "largest (never the minimum or the maximum, and none for a number named as a date, such as `dob`); the "
                "most frequent values of the other columns with their counts, only values shared by at least 5 rows, "
                "and none for free text or an identifier (more than 30 values, most of them rare: a name, a "
                "participant code or a date of birth written on each visit of more than 30 people); the first and "
                "last year of date columns, and of numbers "
                "written as dates (19620115). Rows are counted, not people: a value written on 5 visits of one person "
                "is sent, and so is a name in a study of 30 people or fewer."
            ),
            "- **No row of your dataset.**",
            (
                "- **Data > Clean, when you ask the AI engine:** the name of the variable and its values as written, "
                "only values held by at least 5 rows (rows, not people, as above), without their counts; never free "
                "text or an identifier (a name, a participant code or a date of birth written on each visit of more "
                "than 30 people)."
            ),
            (
                "- **What you write and choose:** your descriptions and criteria, the text of the PDFs you index "
                "(each passage with its PDF's name), the names of the PDFs you ask Document insight about, and your "
                "chat messages; never the folder a PDF is in."
            ),
        ]
        provider = cfg["provider"]
        if provider in file_readers():  # its connector's read_document
            lines.append(
                "- **Document insight:** the PDF file itself while its graph is extracted, with the details it holds "
                f"(such as its title and author) but not its file name; the app then asks {provider_label(provider)} "
                "to delete it."
            )
        else:  # manage.rag_documents: the first 30 pages that have text, cut at 35,000 characters
            lines.append(
                "- **Document insight:** the text of the first 30 pages that have text of each PDF whose graph you "
                "extract, 35,000 characters at most, whether the PDF is indexed or not."
            )
    if cfg.get("provider") == "ollama" and (cloud_model(cfg.get("model")) or cfg.get("remote_host")):
        lines.append(
            "- **An Ollama cloud model** runs on ollama.com under the account signed in on this server's Ollama, the "
            "same for every user of this server."
        )
    if embedding in recipients:
        lines.append(
            f"- **{provider_label(embedding)}, for the embeddings:** the text of the PDFs you index and every "
            "search text (your questions, column names and hints)."
        )
    if store in recipients:
        lines.append(
            f"- **{provider_label(store)}, your vector database:** the text of the PDFs you index (each passage with "
            "its PDF's name, never the folder it is in) and its vectors, kept there until the index is rebuilt or "
            "deleted on that server, and the vector of each search."
        )
    return (
        "\n".join(lines)
        + "\n\nThis confirmation is yours, for this browser tab only. What was already sent stays with each "
        "recipient, under its own terms. It does not replace the approvals of your data-protection framework: do not "
        "give it for identifiable patient data unless your approval covers it."
    )


def _engine_recipients(manager) -> tuple[list[str], str | None]:
    """The recipients the running engine sends data to, by role as it uses them (never by position: a Qdrant server
    is no embedding provider), and its embedding provider among them."""
    cfg = {
        "provider": manager.provider,
        "base_url": manager.base_url,
        "model": getattr(manager, "model_name", None),
        "embedding_provider": manager.embedding_provider,
        "embedding_base_url": manager.embedding_base_url,
        "embedding": getattr(manager, "embedding_model", None),
        "remote_host": getattr(manager, "model_remote_host", None),
    }
    chat, embedding, store, server = _roles(
        cfg, {"vector_engine": VectorEngine.QDRANT, "qdrant_url": manager.qdrant_url}
    )
    return [r for r in (chat, server, embedding, store) if r], embedding


def _running_engine(manager) -> None:
    """Where the running engine sends data, whatever provider is selected below, and how to stop it at once."""
    if not manager.initialized or not manager.recipients():
        return
    order, embedding = _engine_recipients(manager)
    st.info(f"The AI engine runs with {_names(order, embedding)}: your data goes there.", icon=":material/cloud_sync:")
    if st.button("Withdraw the confirmation and stop", key="rag_withdraw_stop", icon=":material/block:"):
        withdrawn = manager.recipients()
        manager.withdraw_consent(*withdrawn)
        for key in [k for k in st.session_state if isinstance(k, str) and k.startswith("rag_consent::")]:
            if set(_consent_recipients(key)) & withdrawn:
                st.session_state[key] = False  # the boxes of these recipients start unticked
        st.session_state["rag_stop_notice"] = STOP_NOTICE
        st.rerun()


def _consent_card(cfg: dict, vector: dict | None = None) -> bool:
    """
    Explicit, session-scoped confirmation before any data is sent outside this machine and the private network
    (audit E6). One box per set of recipients, naming each (with Claude, a cloud embedding provider too; a Qdrant
    server there that keeps the index); ticking or unticking it is what counts, and unticking stops an engine that
    sends data to one of them.
    """
    manager = st.session_state.rag_manager
    with st.container(border=True):
        st.subheader("Data sharing", icon=":material/shield_person:", anchor=False)
        if notice := st.session_state.pop("rag_stop_notice", None):
            st.toast(notice, icon=":material/block:")
        _running_engine(manager)
        recipients = _recipients(cfg, vector)
        # choosing a local provider changes nothing for a cloud engine that runs until another starts
        url = cfg.get("embedding_base_url") or cfg.get("base_url") or ""
        on_qdrant = (vector or {}).get("vector_engine") == VectorEngine.QDRANT
        store = (vector or {}).get("qdrant_url") if on_qdrant else None
        elsewhere = bool(store) and not on_this_machine(store)  # a Qdrant server on the private network
        if not recipients and cfg.get("where_unknown"):  # the server did not say where it runs the model
            model = as_code(str(cfg.get("model") or "the model"))
            said = (
                f"Where the Ollama server runs {model} is not known. Nothing is sent until it says where; if it runs "
                "it elsewhere (ollama.com), you are asked to confirm here first."
            )
            if not on_this_machine(url):
                said += (
                    f" Ollama at {host_of(url)}, on your network, then receives the prompts, the text of your "
                    "documents and the column summaries."
                )
            st.caption(said)
            if elsewhere:
                st.caption(f"The index of your documents is kept on the Qdrant server at {host_of(store)}, on your network.")
            return True
        if not recipients:
            if on_this_machine(url):
                kept = "every prompt and dataset excerpt" if elsewhere else "every prompt, document and dataset excerpt"
                st.caption(f"The local provider keeps {kept} on this server.")
            else:
                st.caption(
                    f"Ollama at {host_of(url)}, on your network, receives the prompts, the text of your documents "
                    "and the column summaries."
                )
            if elsewhere:
                st.caption(f"The index of your documents is kept on the Qdrant server at {host_of(store)}, on your network.")
            return True
        names = _names(recipients, _roles(cfg, vector)[1])
        st.warning(
            _notice(recipients, cfg, vector),
            title=f"What leaves this server with {names}",
            icon=":material/cloud_upload:",
        )
        key = _consent_key(recipients)
        if key not in st.session_state and manager.is_confirmed(frozenset(recipients)):
            st.session_state[key] = True  # confirmed in this session, for an engine started before this page
        consent = bool(
            st.checkbox(
                f"I confirm this data may be sent to {names}",
                key=key,
                on_change=_consent_changed,
                args=(key, tuple(recipients)),
                help=f"Required before any of your data is sent to {names}. Applies to this browser tab only; "
                "untick it to stop an engine that uses them.",
                persist_state=PERSIST,
            )
        )
        if consent and not manager.is_confirmed(frozenset(recipients)):
            manager.confirm(*recipients)  # ticked earlier in the session, for an engine manager rebuilt since
        return consent


def _library_card() -> list[str] | None:
    """Upload PDFs into the user's document library, choose the ones to index and remove the others (the
    index is rebuilt from the remaining documents at the next initialisation: ``manage.vectorstores.index_matches``)."""
    with st.container(border=True):
        st.subheader("Document library", icon=":material/library_books:", anchor=False)
        uploaded_files = st.file_uploader(
            "Add literature (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            key="rag_file_uploader",
            help="PDF files are stored in your workspace and indexed into your knowledge base.",
        )
        docs_dir = _documents_dir()
        if uploaded_files:
            os.makedirs(docs_dir, exist_ok=True)
            for uploaded_file in uploaded_files:
                with open(os.path.join(docs_dir, safe_filename(uploaded_file.name)), "wb") as handle:
                    handle.write(uploaded_file.getbuffer())
            st.success(f"Stored {len(uploaded_files)} document(s). Initialise the engine to index them.", icon=":material/check:")
        all_files = sorted(f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")) if os.path.exists(docs_dir) else []
        if not all_files:
            st.caption("No PDF in your library yet. Indexed papers power the Documents page and the AI suggestions.")
            return None
        key = "rag_selected_files_multiselect"
        selected = st.session_state.setdefault(key, all_files)
        if any(f not in all_files for f in selected):
            st.session_state[key] = [f for f in selected if f in all_files]  # a removed document leaves the choice
        chosen = st.multiselect(
            f"Documents to index ({len(all_files)} in your library)",
            options=all_files,
            key=key,
            help="Only the selected documents are embedded when the engine is initialised.",
            persist_state=PERSIST,
        )
        with st.expander("Remove a document"):
            for name in all_files:
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.markdown(f":material/picture_as_pdf: {name}", width="stretch")
                    if st.button(
                        "Remove",
                        key=f"rag_remove_document_{name}",
                        icon=":material/delete:",
                        help="Deletes this PDF from your library.",
                    ):
                        confirm(
                            f"Remove {name}?",
                            "Deletes the PDF from your library. The AI engine keeps its current index until it is "
                            "initialised again; the index is then rebuilt from the remaining documents.",
                            lambda name=name: remove_pdf(docs_dir, name),
                            confirm_label="Remove",
                        )
        return chosen


def _retrieval_card() -> tuple[float, float]:
    with st.container(border=True):
        st.subheader("Retrieval", icon=":material/tune:", anchor=False)
        st.session_state.setdefault("rag_adherence_score", 0.5)
        st.session_state.setdefault("rag_temperature", 0.3)
        adherence = st.slider(
            "Adherence to your documents",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.rag_adherence_score,
            step=0.1,
            key="rag_adherence_slider",
            help="0 = general medical knowledge, 1 = answers strictly grounded in your documents.",
        )
        if adherence != st.session_state.rag_adherence_score:
            st.session_state.rag_adherence_score = adherence
            if st.session_state.rag_manager.initialized:
                ok, msg = st.session_state.rag_manager.update_adherence_score(adherence)
                (st.toast if ok else st.error)(msg)
        temperature = st.slider(
            "Creativity (temperature)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.rag_temperature,
            step=0.1,
            key="rag_temp_slider",
            help="0 = deterministic answers, 1 = more varied wording.",
        )
        st.session_state.rag_temperature = temperature
    return adherence, temperature


@st.cache_data(ttl=300, show_spinner=False)
def _machine(storage: str) -> MachineProfile:
    """CPU, memory and free disk, probed at most every five minutes."""
    return probe_machine(Path(storage))


def _vectors_dir() -> Path:
    scope = get_current_scope()
    return scope.vectors_dir() if scope else Path(get_vectors_dir())


def _vector_card(selected_files: list[str] | None) -> dict:
    """Choice of the vector database for this machine and library; returns the engine settings."""
    settings = VectorStoreSettings.from_env()
    base_dir = _vectors_dir()
    adopt_earlier_index(base_dir / "qdrant")  # an earlier version's server index, to its server's folder
    docs_dir = Path(_documents_dir())
    sizes = [(docs_dir / f).stat().st_size for f in (selected_files or []) if (docs_dir / f).exists()]
    built = [m for m in (IndexManifest.load(base_dir / name) for name in VECTOR_STORES.names()) if m is not None]
    chunks = max([m.chunks for m in built], default=0) or estimate_chunks(sizes)
    dim = built[0].dim if built else 768
    machine = _machine(str(base_dir))
    with st.container(border=True):
        st.subheader("Vector database", icon=":material/database:", anchor=False)
        _seed_choice("rag_qdrant_url", QDRANT_PLACE)
        prior = st.session_state.get("rag_qdrant_url")
        server_url = prior if prior in _options(QDRANT_PLACE) else ""  # a removed server is no server
        advice = recommend(machine, chunks=chunks, dim=dim, qdrant_server=bool(server_url))
        # the saved engine when it can run here, else the server's, else the one recommended for this machine
        chosen = connector_selector(
            VECTOR_STORES,
            key="rag_vector_engine",
            label="Engine",
            preference=preferences.current().vector_store,
            preference_name="vector_store",
            setting=settings.engine,
            offer_auto=False,
            auto=advice.engine.value,
            auto_reasons=advice.reasons,
            widget="segmented",
            persist=PERSIST,
            help_text="Where the chunks of your documents and their vectors are stored and searched.",
            explain=False,  # below the engine's summary
        )
        engine = chosen.chosen or advice.engine.value
        st.caption(VECTOR_STORES.get(engine).summary)
        if engine == advice.engine:
            st.badge("Recommended for this machine", icon=":material/recommend:", color="green")
        else:
            st.badge(f"Recommended here: {VECTOR_STORES.display_name(advice.engine)}", icon=":material/info:", color="gray")

        def this_machine(selection: Selection) -> None:
            if selection.reasons != advice.reasons:  # chosen for another reason: the recommendation, and why
                st.markdown(f"Recommended for this machine: {VECTOR_STORES.display_name(advice.engine)}.")
                for reason in advice.reasons:
                    st.markdown(f"- {reason}")
            memory = f"{machine.memory_gb:.0f} GB memory" if machine.memory_gb else "memory unknown"
            if machine.available_gb:
                memory += f" ({machine.available_gb:.1f} GB free)"
            disk = f"{machine.disk_free_gb:,.0f} GB free disk" if machine.disk_free_gb is not None else "free disk unknown"
            source = "current index" if built else "rough estimate from the PDF sizes"
            st.caption(f"{machine.cpu_count} CPU cores · {memory} · {disk} · library: ~{chunks:,} chunks ({source}).")

        explain_choice(VECTOR_STORES, chosen, details=this_machine)

        qdrant_url = qdrant_key = None
        if engine == VectorEngine.QDRANT:
            server_url = _address_choice(
                "rag_qdrant_url",
                "Qdrant server",
                "Embedded keeps the index in your workspace. A server (Docker, self-hosted or Qdrant Cloud) is shared "
                "by every user, each with a collection of their own.",
                QDRANT_PLACE,
            )
            qdrant_url = server_url  # empty: embedded, as the option says (never the server's setting then)
            if server_url:  # listed with its scheme: the badge never depends on the key
                _address_badge(server_url, "the passages of your PDFs and the API key")
            _manage_addresses("rag_qdrant_url", QDRANT_PLACE)  # below the address's badge
            if server_url:
                qdrant_key = st.text_input(
                    "Qdrant API key", value="", type="password", key=f"rag_qdrant_api_key::{server_url}",
                    icon=":material/key:",
                    help="Only for a server that asks for one. Kept for this session only. Left empty, the deployment's "
                    "server uses the deployment's key; another server gets none.",
                    persist_state=PERSIST,
                ) or settings.key_for(server_url)  # the server's key goes to its own address (CMP_QDRANT_URL) only
                qdrant_url = explicit_address(server_url, qdrant_key)
                scope = get_current_scope()
                # its own folder in the workspace (an earlier version's index moved there): no request is made
                server = QdrantIndex.open(
                    base_dir / "qdrant", username=scope.username if scope else None, url=qdrant_url,
                    api_key=qdrant_key,
                )
                if st.button("Test the connection", key="rag_qdrant_ping", icon=":material/network_check:"):
                    ok, said = server.ping()  # its list of collections, with the key above
                    if ok:
                        st.success(f"Connected to {_shown(server_url)}: {said}. None of your data was sent.")
                    else:
                        st.error(f"Not connected to {_shown(server_url)}: {said}. None of your data was sent.")
                if server.named_after_account:  # read from the workspace: no request is made
                    st.info(
                        "Your index on the Qdrant server was built by an earlier version: its collection is named "
                        "after your account, and each passage gives the folder of its PDF on this server. Rebuild it "
                        "(**Initialise / reindex** without reusing the index): the new one holds neither, and the "
                        "old one is deleted.",
                        icon=":material/history:",
                    )
            else:
                for note in qdrant_embedded_warnings(machine, chunks=chunks, dim=dim):
                    st.caption(f":orange[:material/warning:] {note}")
        for warning in advice.warnings:
            st.warning(warning, icon=":material/storage:")
        stranded = [
            (url, collection, built)
            for url, collection, built in server_indexes(base_dir / "qdrant")
            if not QDRANT_ADDRESSES.offered(url)  # removed by an administrator, or never listed: nothing is sent there
        ]
        for url, collection, built in stranded:
            st.caption(
                f"Your index ({built.chunks:,} chunks, built {_day(built.built_at)}) is on {_shown(url)}, which this "
                "list does not offer. To use it again, ask a CohortStudio administrator to add that address, as "
                f"written here; if it should go instead, ask whoever runs that server to delete the collection "
                f"{_shown(collection)}."
            )
        if leftover := unrecorded_collection(base_dir / "qdrant"):
            st.caption(
                f"An earlier version left a collection named after your account, {preferences.shown(leftover)}, "
                "on a Qdrant server it did not record: delete it there if it still exists."
            )

        # the index of the place in use: a Qdrant server's is kept in a folder of its own
        on_server = engine == VectorEngine.QDRANT and qdrant_url
        manifest = IndexManifest.load(server_folder(base_dir / engine, qdrant_url) if on_server else base_dir / engine)
        if manifest is not None:
            st.caption(
                f":material/inventory_2: Index: {manifest.chunks:,} chunks from {len(manifest.documents)} document(s), "
                f"embedded with {manifest.embedding} ({manifest.dim} dimensions), built {manifest.built_at[:16].replace('T', ' ')}."
            )
        elif stranded:
            st.caption(
                f":material/inventory_2: No index here yet: Initialise / reindex builds one here; your index on "
                f"{_shown(stranded[0][0])} is not used."
            )
        else:
            st.caption(":material/inventory_2: No index for this engine yet: initialise the engine to build it.")
    return {"vector_engine": engine, "qdrant_url": qdrant_url, "qdrant_api_key": qdrant_key}


def _engine_card(
    cfg: dict,
    consent: bool,
    adherence: float,
    temperature: float,
    selected_files: list[str] | None,
    vector: dict,
) -> None:
    manager = st.session_state.rag_manager
    with st.container(border=True):
        st.subheader("AI engine", icon=":material/bolt:", anchor=False)
        if manager.initialized:
            engine_name = VECTOR_STORES.display_name(manager.vector_engine) if getattr(manager, "vector_engine", None) else None
            st.badge(f"Active · {engine_name} index" if engine_name else "Active", icon=":material/check_circle:", color="green")
        elif getattr(manager, "stop_reason", None):
            st.badge("Stopped", icon=":material/block:", color="red")
            st.caption(
                f"The engine stopped because {manager.stop_reason}: nothing more is sent there. Choose another one "
                "above, then Initialise / reindex."
                if ADDRESS_REMOVED in str(manager.stop_reason)
                else f"The engine stopped because {manager.stop_reason}: nothing more is sent from this tab. To resume, "
                "tick the box again, then Initialise / reindex."
            )
        else:
            st.badge("Not initialised", icon=":material/radio_button_unchecked:", color="gray")
        use_existing_db = st.toggle(
            "Reuse the existing index",
            value=True,
            key="rag_use_existing_db",
            help="Uses the index already built with the selected vector database and embedding model (faster), when "
            "it holds the selected documents: after a document is added or removed, the index is rebuilt. Turn it "
            "off after changing the embedding model.",
            persist_state=PERSIST,
        )
        if not consent:
            st.caption("Confirm the data-sharing notice to enable initialisation.")
        if st.button(
            "Initialise / reindex",
            key="rag_init_button",
            type="primary",
            width="stretch",
            icon=":material/rocket_launch:",
            disabled=not consent,
        ):
            with st.status("Initialising the AI engine", expanded=True) as status:
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(percent: int, message: str) -> None:
                    progress_bar.progress(percent)
                    status_text.caption(message)

                dataset = st.session_state.get("data")
                if dataset is None:
                    dataset = st.session_state.get("working_df")
                if cfg.get("api_key"):
                    manager.api_key = cfg["api_key"]
                # the engine checks that every provider it would send data to is confirmed (the card above)
                success, msg = manager.initialize_system(
                    provider=cfg["provider"],
                    model_name=cfg["model"],
                    embedding_model=cfg["embedding"],
                    base_url=cfg.get("base_url"),
                    adherence_score=adherence,
                    temperature=temperature,
                    dataset_columns=dataset.columns.tolist() if dataset is not None else None,
                    use_existing_db=use_existing_db,
                    progress_callback=update_progress,
                    selected_files=selected_files,
                    embedding_provider=cfg.get("embedding_provider"),
                    embedding_api_key=cfg.get("embedding_api_key"),
                    embedding_base_url=cfg.get("embedding_base_url"),
                    vector_engine=vector.get("vector_engine"),
                    qdrant_url=vector.get("qdrant_url"),
                    qdrant_api_key=vector.get("qdrant_api_key"),
                )
                status.update(label=msg if success else "Initialisation failed", state="complete" if success else "error")
                if not success:
                    st.error(plain(msg, limit=500))
        context = getattr(manager, "global_context", None) if manager.initialized else None
        if isinstance(context, dict):
            st.caption("Inferred clinical context")
            st.markdown(f"**{plain(context.get('domain', 'General medical'), limit=200)}**")
            if context.get("context_description"):
                st.caption(plain(context["context_description"], limit=2000))


def render_ai_settings() -> None:
    """AI provider, keys, consent, document library, retrieval parameters and engine (Settings page)."""
    _ensure_manager()
    left, right = st.columns([3, 2], gap="large")
    with left:
        cfg = _provider_card()
        consent_slot = st.container()  # filled once the vector database is known: a Qdrant server is a recipient
        selected_files = _library_card()
    with right:
        adherence, temperature = _retrieval_card()
        vector = _vector_card(selected_files)
    with consent_slot:
        consent = _consent_card(cfg, vector)
    with right:
        _engine_card(cfg, consent, adherence, temperature, selected_files, vector)


__all__ = ["render_ai_settings"]
