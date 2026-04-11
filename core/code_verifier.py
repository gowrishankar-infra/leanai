"""
LeanAI — Code-Grounded Verification
Post-generation fact-checker that verifies claims about the user's code
against the actual AST and project brain.

How it works:
  1. Model generates a response about the user's code
  2. This module extracts all claims about functions, classes, files, imports
  3. Each claim is verified against the project brain's AST data
  4. Wrong claims are flagged with corrections
  5. Verified claims are marked as confirmed

This eliminates hallucinations about the USER'S code — the #1 complaint
about AI coding assistants. Cloud AI literally cannot do this because
it doesn't have the AST.

Example:
  Model says: "The generate() function takes 3 parameters"
  Brain has:  generate(self, query, config=None, project_context="")
  Output:     "✓ generate() exists — actually takes 4 parameters (self, query, config, project_context)"
"""

import re
import os
from typing import Optional, List, Tuple, Dict


class CodeGroundedVerifier:
    """
    Verifies AI responses against the actual project brain.
    
    Usage:
        verifier = CodeGroundedVerifier(brain=project_brain)
        
        response = "The generate function takes 2 arguments..."
        verified = verifier.verify(response)
        # Returns response with corrections appended
    """

    def __init__(self, brain=None):
        self.brain = brain
        self._stats = {
            "total_checks": 0,
            "claims_found": 0,
            "claims_verified": 0,
            "claims_corrected": 0,
            "claims_unverifiable": 0,
        }

    def update_brain(self, brain):
        """Update the brain reference (after /brain scan)."""
        self.brain = brain

    def verify(self, response: str, query: str = "") -> str:
        """
        Verify a response against the project brain.
        Returns the response with a verification section appended if issues found.
        """
        if not self.brain:
            return response

        self._stats["total_checks"] += 1

        # Extract claims from the response
        claims = self._extract_claims(response)
        if not claims:
            return response

        self._stats["claims_found"] += len(claims)

        # Verify each claim
        results = []
        for claim_type, claim_name, claim_detail in claims:
            verification = self._verify_claim(claim_type, claim_name, claim_detail)
            if verification:
                results.append(verification)

        if not results:
            return response

        # Count results
        corrections = [r for r in results if r[0] == "corrected"]
        verified = [r for r in results if r[0] == "verified"]
        not_found = [r for r in results if r[0] == "not_found"]

        self._stats["claims_verified"] += len(verified)
        self._stats["claims_corrected"] += len(corrections)
        self._stats["claims_unverifiable"] += len(not_found)

        # Only append if there are corrections or useful verifications
        if not corrections and not verified:
            return response

        # Build verification section
        section = "\n\n### Code Verification (checked against your project)\n"

        for status, name, detail in results:
            if status == "corrected":
                section += f"\n⚠ **{name}**: {detail}"
            elif status == "verified":
                section += f"\n✓ **{name}**: {detail}"

        if corrections:
            section += f"\n\n_{len(corrections)} claim(s) corrected based on your actual code._"

        return response.rstrip() + section

    def _extract_claims(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract claims about code from a response.
        Returns list of (claim_type, name, detail).
        
        Claim types: function, class, file, import, parameter
        """
        claims = []
        text_lower = text.lower()

        # Pattern: "the X function" or "X() function" or "function X"
        func_patterns = [
            r"(?:the\s+)?[`\"]?(\w{3,})\(\)[`\"]?\s+(?:function|method)",
            r"(?:the\s+)?[`\"]?(\w{3,})[`\"]?\s+(?:function|method)",
            r"(?:function|method|def)\s+[`\"]?(\w{3,})[`\"]?",
            r"[`](\w{3,})\(\)[`]",
        ]
        for pattern in func_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(1)
                # Skip common words that aren't function names
                skip = {"the", "this", "that", "from", "with", "your", "will",
                        "can", "does", "not", "and", "for", "are", "has", "its",
                        "use", "new", "get", "set", "all", "any", "run", "add",
                        "let", "var", "def", "int", "str", "list", "dict", "none",
                        "true", "false", "self", "main", "test", "print", "type",
                        "input", "output", "error", "value", "return", "import"}
                if name.lower() not in skip and len(name) > 2:
                    claims.append(("function", name, ""))

        # Pattern: "the X class" or "class X"
        class_patterns = [
            r"(?:the\s+)?[`\"]?(\w{3,})[`\"]?\s+class",
            r"class\s+[`\"]?(\w{3,})[`\"]?",
        ]
        for pattern in class_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(1)
                if name[0].isupper():  # Classes are typically PascalCase
                    claims.append(("class", name, ""))

        # Pattern: "X takes N parameters/arguments"
        param_patterns = [
            r"[`\"]?(\w{3,})\(\)[`\"]?\s+takes\s+(\d+)\s+(?:parameter|argument)",
            r"[`\"]?(\w{3,})[`\"]?\s+takes\s+(\d+)\s+(?:parameter|argument)",
            r"[`\"]?(\w{3,})[`\"]?\s+(?:accepts|expects|requires)\s+(\d+)\s+(?:parameter|argument)",
        ]
        for pattern in param_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(1)
                count = match.group(2)
                claims.append(("parameter_count", name, count))

        # Pattern: "in file X.py" or "from X.py"
        file_patterns = [
            r"(?:in|from|file)\s+[`\"]?(\w+\.py)[`\"]?",
            r"[`\"](\w+\.py)[`\"]",
        ]
        for pattern in file_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(1)
                claims.append(("file", name, ""))

        # Pattern: "returns X" where X is a type
        return_patterns = [
            r"[`\"]?(\w{3,})\(\)[`\"]?\s+returns?\s+(?:a\s+)?[`\"]?(\w+)[`\"]?",
        ]
        for pattern in return_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(1)
                return_type = match.group(2)
                claims.append(("return_type", name, return_type))

        # Deduplicate
        seen = set()
        unique = []
        for claim in claims:
            key = (claim[0], claim[1])
            if key not in seen:
                seen.add(key)
                unique.append(claim)

        return unique[:10]  # Max 10 claims to avoid excessive checking

    def _verify_claim(self, claim_type: str, name: str, detail: str) -> Optional[Tuple[str, str, str]]:
        """
        Verify a single claim against the brain.
        Returns (status, name, message) or None if not checkable.
        """
        try:
            if claim_type == "function":
                return self._verify_function(name)
            elif claim_type == "class":
                return self._verify_class(name)
            elif claim_type == "parameter_count":
                return self._verify_param_count(name, int(detail))
            elif claim_type == "file":
                return self._verify_file(name)
            elif claim_type == "return_type":
                return self._verify_return_type(name, detail)
        except Exception:
            return None
        return None

    def _verify_function(self, name: str) -> Optional[Tuple[str, str, str]]:
        """Verify a function exists in the project."""
        if not hasattr(self.brain, 'graph') or not hasattr(self.brain.graph, '_function_lookup'):
            return None

        lookup = self.brain.graph._function_lookup
        
        # Exact match
        if name in lookup:
            info = lookup[name]
            if isinstance(info, list) and len(info) > 0:
                loc = info[0]
                file_path = loc.get("file", "unknown")
                args = loc.get("args", "")
                return ("verified", f"{name}()", f"Confirmed — found in {file_path}")
            return ("verified", f"{name}()", "Confirmed — exists in project")

        # Case-insensitive match
        for func_name in lookup:
            if func_name.lower() == name.lower():
                return ("corrected", f"{name}()", f"Name is actually `{func_name}()` (case mismatch)")

        # Not found — might be from standard library, don't flag
        return None

    def _verify_class(self, name: str) -> Optional[Tuple[str, str, str]]:
        """Verify a class exists in the project."""
        if not hasattr(self.brain, 'graph') or not hasattr(self.brain.graph, '_class_lookup'):
            return None

        lookup = self.brain.graph._class_lookup

        if name in lookup:
            info = lookup[name]
            if isinstance(info, list) and len(info) > 0:
                loc = info[0]
                file_path = loc.get("file", "unknown")
                return ("verified", f"class {name}", f"Confirmed — found in {file_path}")
            return ("verified", f"class {name}", "Confirmed — exists in project")

        # Case-insensitive
        for cls_name in lookup:
            if cls_name.lower() == name.lower():
                return ("corrected", f"class {name}", f"Name is actually `{cls_name}` (case mismatch)")

        return None

    def _verify_param_count(self, func_name: str, claimed_count: int) -> Optional[Tuple[str, str, str]]:
        """Verify the parameter count of a function."""
        if not hasattr(self.brain, 'graph') or not hasattr(self.brain.graph, '_function_lookup'):
            return None

        lookup = self.brain.graph._function_lookup

        # Find the function
        func_info = None
        actual_name = func_name
        for name in lookup:
            if name.lower() == func_name.lower():
                if isinstance(lookup[name], list) and len(lookup[name]) > 0:
                    func_info = lookup[name][0]
                    actual_name = name
                break

        if not func_info:
            return None

        # Count parameters from args string
        args_str = func_info.get("args", "")
        if not args_str:
            return None

        # Parse parameter count
        params = [a.strip() for a in args_str.split(",") if a.strip()]
        actual_count = len(params)

        if actual_count == claimed_count:
            return ("verified", f"{actual_name}()", f"Confirmed — takes {actual_count} parameters: {args_str}")
        else:
            return ("corrected", f"{actual_name}()",
                    f"Actually takes {actual_count} parameters ({args_str}), not {claimed_count}")

    def _verify_file(self, filename: str) -> Optional[Tuple[str, str, str]]:
        """Verify a file exists in the project."""
        if not hasattr(self.brain, '_file_analyses'):
            return None

        # Check if file exists in brain
        for rel_path in self.brain._file_analyses:
            if os.path.basename(rel_path) == filename or rel_path.endswith(filename):
                analysis = self.brain._file_analyses[rel_path]
                func_count = len(analysis.get("functions", []))
                class_count = len(analysis.get("classes", []))
                return ("verified", filename,
                        f"Confirmed — {func_count} functions, {class_count} classes")

        # File not in brain — might exist but not indexed
        return None

    def _verify_return_type(self, func_name: str, claimed_type: str) -> Optional[Tuple[str, str, str]]:
        """Verify the return type of a function."""
        if not hasattr(self.brain, 'graph') or not hasattr(self.brain.graph, '_function_lookup'):
            return None

        lookup = self.brain.graph._function_lookup

        for name in lookup:
            if name.lower() == func_name.lower():
                if isinstance(lookup[name], list) and len(lookup[name]) > 0:
                    info = lookup[name][0]
                    actual_return = info.get("returns", "")
                    if actual_return:
                        if claimed_type.lower() in actual_return.lower():
                            return ("verified", f"{name}()",
                                    f"Confirmed — returns {actual_return}")
                        else:
                            return ("corrected", f"{name}()",
                                    f"Actually returns `{actual_return}`, not `{claimed_type}`")
                break

        return None

    def should_verify(self, query: str, response: str) -> bool:
        """Determine if a response should be verified."""
        if not self.brain:
            return False

        # Only verify responses that discuss the user's project
        project_indicators = [
            "function", "class", "method", "file", "module",
            "takes", "returns", "parameter", "argument",
            "import", "define", "implement",
        ]
        response_lower = response.lower()
        return any(ind in response_lower for ind in project_indicators)

    def stats(self) -> dict:
        return dict(self._stats)
