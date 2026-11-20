# Source data for the voice details

## `vctk-speaker-info.txt`

This is `speaker-info.txt` from `VCTK-Corpus-0.92.zip` on Edinburgh DataShare
(https://datashare.ed.ac.uk/handle/10283/3443), unmodified. Its sha256 is
`ee6447edab776828e0c2e726c04a4965c4602c32047b14fac2f5294d18728eee` (4,028 bytes).

`tools/convert/voice_details.py` reads it to describe the `en-vctk` voice's speakers (gender, age, accent and
region) in `src/bytestts/catalog/details.json`. The tool refuses the file if its sha256 differs, so it is kept with
LF line endings (`.gitattributes` marks this folder `-text`).

Credit: CSTR VCTK Corpus (version 0.92), Junichi Yamagishi, Christophe Veaux and Kirsten MacDonald, The Centre for
Speech Technology Research, University of Edinburgh (2019), https://doi.org/10.7488/ds/2645, CC BY 4.0.

The file is shared under the same licence, Creative Commons Attribution 4.0 International
(https://creativecommons.org/licenses/by/4.0/). The descriptions in `details.json` are adapted from it: the gender is
spelled out (`F` as Female, `M` as Male), accent names are split into words (`NorthernIrish` as Northern Irish),
`English` is shown as `English (England)`, and `Unknown` accents and the comments in parentheses are left out.
