# Technical lists

Lists whose items hold code, file paths, command-line flags, package names, version numbers, URLs or identifiers: setup steps, a deploy checklist, files to change, packages to install, config keys to set. The items keep every code token exactly, with spoken symbols written as symbols.

Mix ordered steps ("1.", "2.") and unordered items ("- "). Include a lead-in taken from the speaker's words.

Layout is "list" or "mixed".

Tags: setup_steps, files_to_change, packages, config_keys, deploy_checklist.

Example transcript (raw style): "ok deploy steps first run npm run build then uh copy the dist folder to slash var slash www and then restart nginx with sudo systemctl restart nginx"
Cleaned: "Deploy steps:
1. Run npm run build.
2. Copy the dist folder to /var/www.
3. Restart nginx with sudo systemctl restart nginx."
