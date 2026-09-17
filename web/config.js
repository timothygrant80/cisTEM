// cisTEM3 -- local configuration.
//
// By default cistem3.html talks to the server that served it: open
// http://<server>:8000/ from any machine and the page uses
// http://<server>:8000/api. Opened directly as a file:// URL it uses
// http://localhost:8000/api instead. Neither needs this file.
//
// To pin a different API address -- the page served from one machine, the
// API on another, or behind a proxy under a different path -- uncomment
// apiBase below, edit it, then reload the page. No other file needs to
// change. If this file is missing, or fails to load (e.g. blocked when the
// page is opened via file:// in some browsers), the defaults above apply.

window.CRYOEM_CONFIG = {
  // apiBase: "http://my-server.example.org:8000/api"
};
