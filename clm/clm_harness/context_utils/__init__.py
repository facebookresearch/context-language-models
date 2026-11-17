# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Context-management utilities for clm_harness.

Modules (import the leaf directly; this package file stays import-light):
* ``context_string`` — render the message log to an editable string and map an
                       arbitrarily-edited string back to a legal message list
                       (``render_editable`` / ``reconcile_editable``; legacy
                       text-only imports use ``parse_back``).
"""
