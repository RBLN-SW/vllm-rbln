# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import re
import sys
from enum import Enum

MIN_YEAR = 2025
CURRENT_YEAR = datetime.date.today().year


def _year_ok(year):
    return MIN_YEAR <= int(year) <= CURRENT_YEAR


# Matches the Rebellions copyright line with any 4-digit year. The trailing
# "All rights reserved." is optional so that older one-line variants are still
# recognized as a copyright line (year validity is checked separately).
COPYRIGHT_RE = re.compile(r"#\s*Copyright\s+(?P<year>\d{4})\s+Rebellions Inc\.")

LICENSE_MARKER = '# Licensed under the Apache License, Version 2.0 (the "License");'


def _copyright_line(year):
    return f"# Copyright {year} Rebellions Inc. All rights reserved.\n"


# The full Apache license body in the canonical Rebellions format, as a list of
# physical lines (each terminated with "\n"). Used when auto-adding the body.
LICENSE_BODY_LINES = [
    '# Licensed under the Apache License, Version 2.0 (the "License");\n',
    "# you may not use this file except in compliance with the License.\n",
    "# You may obtain a copy of the License at:\n",
    "#\n",
    "#     http://www.apache.org/licenses/LICENSE-2.0\n",
    "#\n",
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
    NON_CANONICAL = "non_canonical"  # Complete header with inconsistent formatting
    INVALID_LICENSE = "invalid_license"  # Incomplete or misplaced license body
    WRONG_YEAR = "wrong_year"  # Copyright year outside the allowed range


def check_license_header_status(file_path):
    """Classify the Rebellions license header status of a file."""
    with open(file_path, encoding="UTF-8") as file:
        lines = file.readlines()

    if not lines:
        return LicenseStatus.EMPTY

    header_end = 0
    while header_end < len(lines) and (
        not lines[header_end].strip() or lines[header_end].lstrip().startswith("#")
    ):
        header_end += 1

    copyright_index = None
    license_index = None
    for i, line in enumerate(lines[:header_end]):
        match = COPYRIGHT_RE.match(line.strip())
        if match:
            if not _year_ok(match.group("year")):
                return LicenseStatus.WRONG_YEAR
            copyright_index = i
        if line.strip() == LICENSE_MARKER:
            license_index = i

    if license_index is not None:
        body_end = _license_body_end(lines, license_index)
        if body_end is None:
            return LicenseStatus.INVALID_LICENSE
        if copyright_index is None:
            return LicenseStatus.MISSING_COPYRIGHT
        if copyright_index >= license_index or any(
            line.strip() not in ("", "#")
            for line in lines[copyright_index + 1 : license_index]
        ):
            return LicenseStatus.INVALID_LICENSE
        match = COPYRIGHT_RE.match(lines[copyright_index].strip())
        block = [_copyright_line(match.group("year")), "#\n", *LICENSE_BODY_LINES]
        if lines[copyright_index:body_end] != block:
            return LicenseStatus.NON_CANONICAL
        return LicenseStatus.COMPLETE
    if copyright_index is not None:
        return LicenseStatus.MISSING_LICENSE
    return LicenseStatus.MISSING_BOTH


def _license_body_end(lines, start):
    """Recognize the full body, allowing legacy separator and URL-intro styles."""
    for offset, expected in enumerate(LICENSE_BODY_LINES):
        if start + offset >= len(lines):
            return None
        actual = lines[start + offset].strip()
        if expected.strip() == "#":
            if actual not in ("", "#"):
                return None
        elif actual != expected.strip() and (
            offset != 2 or actual != expected.strip().removesuffix(":")
        ):
            return None
    return start + len(LICENSE_BODY_LINES)


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
        block = [_copyright_line(CURRENT_YEAR), "#\n", *LICENSE_BODY_LINES]
        lines[idx:idx] = block

    elif status == LicenseStatus.MISSING_LICENSE:
        # Insert the license body right after the existing copyright line.
        for i, line in enumerate(lines):
            if COPYRIGHT_RE.match(line.strip()):
                end = i + 1
                while end < len(lines) and lines[end].strip() in ("", "#"):
                    end += 1
                lines[i + 1 : end] = ["#\n", *LICENSE_BODY_LINES, "\n"]
                break

    elif status == LicenseStatus.MISSING_COPYRIGHT:
        # Insert the copyright line just before the license body.
        for i, line in enumerate(lines):
            if line.strip() == LICENSE_MARKER:
                lines[i:i] = [_copyright_line(CURRENT_YEAR), "#\n"]
                end = _license_body_end(lines, i + 2)
                lines[i + 2 : end] = LICENSE_BODY_LINES
                break

    elif status == LicenseStatus.NON_CANONICAL:
        for i, line in enumerate(lines):
            match = COPYRIGHT_RE.match(line.strip())
            if match:
                start = next(
                    j
                    for j in range(i + 1, len(lines))
                    if lines[j].strip() == LICENSE_MARKER
                )
                end = _license_body_end(lines, start)
                lines[i:end] = [
                    _copyright_line(match.group("year")),
                    "#\n",
                    *LICENSE_BODY_LINES,
                ]
                break

    else:
        return

    with open(file_path, "w", encoding="UTF-8") as file:
        file.writelines(lines)


def main():
    fixed = {
        LicenseStatus.MISSING_BOTH: [],
        LicenseStatus.MISSING_COPYRIGHT: [],
        LicenseStatus.MISSING_LICENSE: [],
        LicenseStatus.NON_CANONICAL: [],
    }
    wrong_year = []
    invalid_license = []

    for file_path in sys.argv[1:]:
        status = check_license_header_status(file_path)
        if status in fixed:
            fixed[status].append(file_path)
            add_header(file_path, status)
        elif status == LicenseStatus.WRONG_YEAR:
            wrong_year.append(file_path)
        elif status == LicenseStatus.INVALID_LICENSE:
            invalid_license.append(file_path)

    failed = any(fixed.values()) or wrong_year or invalid_license

    if any(fixed.values()):
        print(
            "The following files had missing or non-canonical RBLN license headers "
            "(auto-fixed; please review and re-stage):"
        )
        for paths in fixed.values():
            for file_path in paths:
                print(f"  {file_path}")

    if wrong_year:
        print(
            f"The following files have a copyright year outside "
            f"{MIN_YEAR}-{CURRENT_YEAR} (fix the year manually):"
        )
        for file_path in wrong_year:
            print(f"  {file_path}")

    if invalid_license:
        print(
            "The following files have incomplete or misplaced Apache license bodies "
            "(fix manually):"
        )
        for file_path in invalid_license:
            print(f"  {file_path}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
