// cisTEM3 -- local configuration.
//
// port: the port the server listens on. `python cistem_server.py` reads it
// from this file at start-up (the CISTEM_PORT environment variable overrides
// it), and cistem3.html uses it when opened directly as a file:// URL, where
// it connects to http://localhost:<port>/api. Served by the server, the page
// simply talks to the server that served it, whatever the port.
//
// apiBase: pins a different API address altogether -- the page served from
// one machine, the API on another, or behind a proxy under another path.
// Uncomment and edit, then reload the page. If this file is missing, or
// fails to load, the defaults (port 8000, same-origin API) apply.

window.CRYOEM_CONFIG = {
  port: 8000,
  // apiBase: "http://my-server.example.org:8000/api"
};
