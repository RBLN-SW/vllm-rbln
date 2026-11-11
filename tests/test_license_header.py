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

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.pre_commit import check_license_header as checker


class TestLicenseHeader(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "sample.py"
        self.header = (
            checker._copyright_line(2025) + "#\n" + "".join(checker.LICENSE_BODY_LINES)
        )

    def test_format_variants_are_fixed_and_idempotent(self):
        prefix = "#!/usr/bin/env python\n# SPDX-License-Identifier: Apache-2.0\n"
        suffix = "\n# Application comment.\nvalue = 1\n"
        for variant in (
            self.header.replace("#\n", "\n"),
            self.header.replace("License at:", "License at"),
            self.header.replace("#\n", "#  \n"),
            self.header.replace("#\n", "\n", 1),
        ):
            with self.subTest(variant=variant):
                self.path.write_text(prefix + variant + suffix)
                status = checker.check_license_header_status(self.path)
                self.assertEqual(status, checker.LicenseStatus.NON_CANONICAL)
                checker.add_header(self.path, status)
                expected = prefix + self.header + suffix
                self.assertEqual(self.path.read_text(), expected)
                status = checker.check_license_header_status(self.path)
                self.assertEqual(status, checker.LicenseStatus.COMPLETE)
                checker.add_header(self.path, status)
                self.assertEqual(self.path.read_text(), expected)

    def test_incomplete_and_corrupted_bodies_are_rejected_without_rewriting(self):
        for body in (
            self.header.split("# Unless required")[0],
            self.header.replace("# WITHOUT WARRANTIES", "# WITH WARRANTIES"),
            self.header.replace("# you may", "you may"),
        ):
            with self.subTest(body=body):
                self.path.write_text(body)
                result = subprocess.run(
                    [sys.executable, checker.__file__, str(self.path)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn("fix manually", result.stdout)
                self.assertEqual(self.path.read_text(), body)

    def test_cli_fixes_blank_separators_and_fails_until_rerun(self):
        self.path.write_text(self.header.replace("#\n", "\n"))
        command = [sys.executable, checker.__file__, str(self.path)]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 1)
        self.assertEqual(self.path.read_text(), self.header)
        second = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(second.returncode, 0)

    def test_missing_parts_are_added_with_commented_separators(self):
        for content in (
            "value = 1\n",
            checker._copyright_line(2025) + "\nvalue = 1\n",
            "".join(checker.LICENSE_BODY_LINES).replace("#\n", "\n") + "\nvalue = 1\n",
        ):
            with self.subTest(content=content):
                self.path.write_text(content)
                checker.add_header(
                    self.path, checker.check_license_header_status(self.path)
                )
                self.assertEqual(
                    checker.check_license_header_status(self.path),
                    checker.LicenseStatus.COMPLETE,
                )
                fixed = self.path.read_text()
                self.assertIn("#\n# Licensed under", fixed)
                self.assertIn(
                    "#\n#     http://www.apache.org/licenses/LICENSE-2.0\n#\n", fixed
                )
                self.assertTrue(fixed.endswith("value = 1\n"))

    def test_body_before_copyright_is_rejected(self):
        self.path.write_text(
            "".join(checker.LICENSE_BODY_LINES) + checker._copyright_line(2025)
        )
        self.assertEqual(
            checker.check_license_header_status(self.path),
            checker.LicenseStatus.INVALID_LICENSE,
        )

    def test_year_must_be_between_project_start_and_current_year(self):
        with patch.object(checker, "CURRENT_YEAR", 2026):
            for year, expected in (
                (2024, checker.LicenseStatus.WRONG_YEAR),
                (2025, checker.LicenseStatus.COMPLETE),
                (2026, checker.LicenseStatus.COMPLETE),
                (2027, checker.LicenseStatus.WRONG_YEAR),
            ):
                with self.subTest(year=year):
                    self.path.write_text(self.header.replace("2025", str(year)))
                    self.assertEqual(
                        checker.check_license_header_status(self.path), expected
                    )

    def test_cli_rejects_future_year_without_rewriting(self):
        content = self.header.replace("2025", str(checker.CURRENT_YEAR + 1))
        self.path.write_text(content)
        result = subprocess.run(
            [sys.executable, checker.__file__, str(self.path)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("fix the year manually", result.stdout)
        self.assertEqual(self.path.read_text(), content)
