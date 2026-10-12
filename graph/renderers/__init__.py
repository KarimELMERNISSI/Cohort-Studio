"""
Rendering backends. Import a backend module explicitly (``graph.renderers.plotly``) or let
:mod:`graph.manager` pick one; this package deliberately does not import any backend eagerly so
that an uninstalled optional dependency (yFiles) never breaks the others.
"""
