"""Study dossier page: the participant flow, the results added to the report and the Word dossier."""

from app_pages.study_dossier import app
from ui.components import page_header

page_header(
    "Study dossier",
    "The participant flow, the results you added to the report and the Word dossier that documents them.",
    chapter="08_Reproduction",
)
app()
