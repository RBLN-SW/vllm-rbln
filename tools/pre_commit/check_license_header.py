# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import re
import sys
from enum import Enum

# Minimum valid copyright year (the project's first year). Any year from this
# onward is accepted -- we intentionally do NOT cap at the current year, so the
# check never depends on the system clock and never rejects a "future" year.
MIN_YEAR = 2025

# Year stamped on a header auto-added to a file that has none. Defaults to the
# current year, but any year >= MIN_YEAR is valid, so this is only a sensible
# default and is never enforced as an upper bound.
NEW_HEADER_YEAR = datetime.date.today().year


def _year_ok(year):
    return int(year) >= MIN_YEAR


# Matches the Rebellions copyright line with any 4-digit year. The trailing
# "All rights reserved." is optional so that older one-line variants are still
# recognized as a copyright line (year validity is checked separately).
COPYRIGHT_RE = re.compile(r"#\s*Copyright\s+(?P<year>\d{4})\s+Rebellions Inc\.")

# A single representative line from the Apache license body. We detect the body
# by this marker rather than by an exact multi-line match, so files that differ
# only in separator style (blank line vs. "#") are still treated as complete.
LICENSE_MARKER = '# Licensed under the Apache License, Version 2.0 (the "License");'


def _copyright_line(year):
    return f"# Copyright {year} Rebellions Inc. All rights reserved.\n"


# The full Apache license body in the canonical Rebellions format, as a list of
# physical lines (each terminated with "\n"). Used when auto-adding the body.
LICENSE_BODY_LINES = [
    '# Licensed under the Apache License, Version 2.0 (the "License");\n',
    "# you may not use this file except in compliance with the License.\n",
    "# You may obtain a copy of the License at:\n",
    "\n",
    "#     http://www.apache.org/licenses/LICENSE-2.0\n",
    "\n",
    "# Unless required by applicable law or agreed to in writing, software\n",
    '# distributed under the License is distributed on an "AS IS" BASIS,\n',
    "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
    "# See the License for the specific language governing permissions and\n",
    "# limitations under the License.\n",
]


class LicenseStatus(Enum):
    """Status of the Rebellions license header in a file."""

    EMPTY = "empty"  # Empty file like __init__.py
    COMPLETE = "complete"  # Valid copyright line + license body
    MISSING_LICENSE = "missing_license"  # Has copyright, missing body
    MISSING_COPYRIGHT = "missing_copyright"  # Has body, missing copyright
    MISSING_BOTH = "missing_both"  # Neither present
    WRONG_YEAR = "wrong_year"  # Copyright present but year not allowed


def check_license_header_status(file_path):
    """Classify the Rebellions license header status of a file."""
    with open(file_path, encoding="UTF-8") as file:
        lines = file.readlines()

    if not lines:
        return LicenseStatus.EMPTY

    has_copyright = False
    copyright_year_ok = False
    has_license_body = False

    for line in lines:
        stripped = line.strip()
        match = COPYRIGHT_RE.match(stripped)
        if match:
            has_copyright = True
            if _year_ok(match.group("year")):
                copyright_year_ok = True
        if stripped == LICENSE_MARKER:
            has_license_body = True

    if has_copyright and not copyright_year_ok:
        return LicenseStatus.WRONG_YEAR

    if copyright_year_ok and has_license_body:
        return LicenseStatus.COMPLETE
    elif copyright_year_ok and not has_license_body:
        return LicenseStatus.MISSING_LICENSE
    elif not copyright_year_ok and has_license_body:
        return LicenseStatus.MISSING_COPYRIGHT
    else:
        return LicenseStatus.MISSING_BOTH


def _header_insert_index(lines):
    """Index at which a fresh header block should be inserted: after a shebang
    and after any leading SPDX lines, matching the repo convention of placing
    the Rebellions block right below the SPDX lines."""
    idx = 0
    if lines and lines[0].startswith("#!"):
        idx = 1
    while idx < len(lines) and lines[idx].lstrip().startswith("# SPDX"):
        idx += 1
    return idx


def add_header(file_path, status):
    """Add or supplement the Rebellions license header based on status."""
    with open(file_path, encoding="UTF-8") as file:
        lines = file.readlines()

    if status == LicenseStatus.MISSING_BOTH:
        idx = _header_insert_index(lines)
        block = [_copyright_line(NEW_HEADER_YEAR), "\n", *LICENSE_BODY_LINES]
        lines[idx:idx] = block

    elif status == LicenseStatus.MISSING_LICENSE:
        # Insert the license body right after the existing copyright line.
        for i, line in enumerate(lines):
            if COPYRIGHT_RE.match(line.strip()):
                lines[i + 1 : i + 1] = ["\n", *LICENSE_BODY_LINES]
                break

    elif status == LicenseStatus.MISSING_COPYRIGHT:
        # Insert the copyright line just before the license body.
        for i, line in enumerate(lines):
            if line.strip() == LICENSE_MARKER:
                lines[i:i] = [_copyright_line(NEW_HEADER_YEAR), "\n"]
                break

    else:
        return  # COMPLETE / EMPTY / WRONG_YEAR are not auto-fixed.

    with open(file_path, "w", encoding="UTF-8") as file:
        file.writelines(lines)


def main():
    fixed = {
        LicenseStatus.MISSING_BOTH: [],
        LicenseStatus.MISSING_COPYRIGHT: [],
        LicenseStatus.MISSING_LICENSE: [],
    }
    wrong_year = []

    for file_path in sys.argv[1:]:
        status = check_license_header_status(file_path)
        if status in fixed:
            fixed[status].append(file_path)
            add_header(file_path, status)
        elif status == LicenseStatus.WRONG_YEAR:
            wrong_year.append(file_path)

    failed = any(fixed.values()) or wrong_year

    if any(fixed.values()):
        print(
            "The following files were missing the RBLN License header "
            "(auto-added; please review and re-stage):"
        )
        for paths in fixed.values():
            for file_path in paths:
                print(f"  {file_path}")

    if wrong_year:
        print(
            f"The following files have a copyright year earlier than "
            f"{MIN_YEAR} (fix the year manually):"
        )
        for file_path in wrong_year:
            print(f"  {file_path}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
