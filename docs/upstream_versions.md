# Upstream Versions

The integration uses these immutable upstream revisions:

- KV-Cloak / Shadow: `6b40f36edb2f337557543e7e60b10022308883d4`
- KIVI: `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`

KIVI is kept as the `third_party/KIVI` git submodule. Its source is not
copied into the project and the integration imports the public functions from
`quant.new_pack` directly.
