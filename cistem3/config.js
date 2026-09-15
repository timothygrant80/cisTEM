// cisTEM3 -- local configuration.
//
// Edit apiBase below to change the pipeline API address cistem3.html
// connects to on load, then reload the page. No other file needs to change.
//
// If this file is missing, or fails to load (e.g. blocked when the page is
// opened via file:// in some browsers), cistem3.html falls back to
// http://localhost:8000/api automatically.

window.CRYOEM_CONFIG = {
  apiBase: "http://localhost:8000/api"
};
