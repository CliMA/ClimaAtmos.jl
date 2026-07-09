#!/usr/bin/env python3
"""Copy a ClimaParams toml file and override one or more `[key]\\nvalue = ...` entries.

Usage:
  set_toml_value.py <base_toml> <out_toml> key1=val1 [key2=val2 ...]

Example:
  set_toml_value.py toml/prognostic_edmfx_1M.toml /tmp/sweep/run1.toml \\
      detr_massflux_vertdiv_coeff=0.1 fixed_snow_terminal_velocity=0.25
"""
import re
import sys


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    base_toml, out_toml = sys.argv[1], sys.argv[2]
    overrides = sys.argv[3:]

    with open(base_toml) as f:
        content = f.read()

    for kv in overrides:
        key, val = kv.split("=", 1)
        pattern = re.compile(r"(\[" + re.escape(key) + r"\]\nvalue = )([^\n]+)")
        content, n = pattern.subn(lambda m: m.group(1) + val, content)
        if n != 1:
            sys.exit(f"ERROR: key '{key}' not found exactly once in {base_toml} (found {n})")

    with open(out_toml, "w") as f:
        f.write(content)
    print(f"wrote {out_toml}")


if __name__ == "__main__":
    main()
