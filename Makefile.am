ACLOCAL_AMFLAGS = -I m4
SUBDIRS = src

# cisTEM3 web front end and its Flask reference server.  `make install` puts
# the tree under $(pkgdatadir)/web (share/cistem/web by default), preserving
# the layout because cistem_server.py serves cistem3.html from its parent
# directory.  Files are listed explicitly rather than copied wholesale so a
# developer's runtime state (server/data/, __pycache__/) never gets installed.
# Add new web files here; developer-only files (CLAUDE.md, .gitignore,
# server/tests/) are deliberately left out.
nobase_dist_pkgdata_DATA = \
	web/abinitio-example.png \
	web/cistem3.html \
	web/class2d-example-1.png \
	web/class2d-example-2.png \
	web/class2d-example-3.png \
	web/class2d-example-4.png \
	web/config.js \
	web/ctffind-definitions.png \
	web/ctffind-diagnostic-image.png \
	web/ctffind-example-1dfit.png \
	web/docs/job-protocol.md \
	web/logo.png \
	web/movie-alignment-example.png \
	web/README.md \
	web/reference/dashboard.html \
	web/refine3d-strategy.png \
	web/server/abinitio.py \
	web/server/auth.py \
	web/server/autorefine.py \
	web/server/cistem_server.py \
	web/server/classification.py \
	web/server/db.py \
	web/server/display.py \
	web/server/generate3d.py \
	web/server/imageheaders.py \
	web/server/job_protocol.py \
	web/server/job_runner.py \
	web/server/package_io.py \
	web/server/preview.py \
	web/server/refine3d.py \
	web/server/refinectf.py \
	web/server/refinement_packages.py \
	web/server/refinements.py \
	web/server/requirements.txt \
	web/server/sharpen.py \
	web/server/stages/ctffind.py \
	web/server/stages/estimate_beamtilt.py \
	web/server/stages/find_particles.py \
	web/server/stages/__init__.py \
	web/server/stages/merge2d.py \
	web/server/stages/merge3d.py \
	web/server/stages/prepare_stack_classaverage.py \
	web/server/stages/prepare_stack.py \
	web/server/stages/reconstruct3d.py \
	web/server/stages/refine2d.py \
	web/server/stages/refine3d.py \
	web/server/stages/refine_ctf.py \
	web/server/stages/unblur.py \
	web/server/starfile.py \
	web/server/symmetry.py \
	web/server/volumes.py \
	web/tools/fake_controller.py
