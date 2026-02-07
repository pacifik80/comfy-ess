import re
import logging
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import random
try:
    from .global_state import GlobalState
except ImportError:
    # For testing purposes
    class GlobalState:
        @staticmethod
        def get_seed():
            return None
        
        @staticmethod
        def get_replacements():
            return {"1girl": "1girl 13yo child, preteen."}

class ParseError(Exception):
    def __init__(self, message: str, line_number: int, line_content: str, context: str = ""):
        self.message = message
        self.line_number = line_number
        self.line_content = line_content
        self.context = context
        super().__init__(f"Parse error at line {line_number}: {message}\nLine: '{line_content}'\n{context}")

@dataclass
class Option:
    name: str
    weight: float
    content: str
    inverse_content: str  # Content that goes to the opposite prompt

class PromptParser:
    def __init__(self, seed: Optional[int] = None):
        # Pattern for option blocks: { ... }
        self.block_pattern = re.compile(r'\{((?:[^{}]|{[^{]|{[^{]|{[^{])*)\}', re.DOTALL)
        # Pattern for option header: [name:weight] or [weight] or [name]
        self.header_pattern = re.compile(r'^\[([^:]+)(?::(\d+(?:\.\d+)?))?\]|\[(\d+(?:\.\d+)?)\]')
        self._seed = self._normalize_seed(seed)
        self._logger = logging.getLogger(__name__)
        self._variant_letters = "abcdefghij"

    @staticmethod
    def _strip_comments(text: str) -> str:
        """
        Remove inline comments started by #.
        Comments end at newline, any of {}|[]:<> or another #.
        """
        if not text:
            return text
        out = []
        in_comment = False
        for ch in text:
            if in_comment:
                if ch == "#":
                    in_comment = False
                    continue
                if ch in "\n{}|[]:<>":
                    in_comment = False
                    out.append(ch)
                    continue
                continue
            if ch == "#":
                in_comment = True
                continue
            out.append(ch)
        return "".join(out)

    @staticmethod
    def _normalize_seed(seed: Optional[int]) -> Optional[int]:
        if seed is None:
            return None
        try:
            value = int(seed)
        except (TypeError, ValueError):
            return None
        if value < 0:
            value = 0
        return value

    def split_prompt(self, text: str) -> Tuple[str, str]:
        """Split text into positive and negative parts based on !> separator.
        Ignore separators that occur inside << >> variant blocks.
        """
        if not text:
            return "", ""
        i = 0
        variant_depth = 0
        while i < len(text) - 1:
            pair = text[i:i+2]
            if pair == "<<":
                variant_depth += 1
                i += 2
                continue
            if pair == ">>":
                if variant_depth > 0:
                    variant_depth -= 1
                i += 2
                continue
            if pair == "!>" and variant_depth == 0:
                pos = text[:i].rstrip()
                neg = text[i+2:].lstrip()
                return pos, neg
            i += 1
        return text.strip(), ""

    def _cleanup_lines(self, text: str) -> str:
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if line and line != ".":
                lines.append(line)
        return "\n".join(lines)

    def _parse_base(self, text: str) -> Tuple[str, str]:
        """Parse text into a resolved template and extra negative content."""
        # Apply replacements from global state if available
        replacements = GlobalState.get_replacements()
        if replacements:
            for key, value in replacements.items():
                text = text.replace(f"%{key}%", value)
                text = text.replace(key, value)

        # Strip inline comments before parsing
        text = self._strip_comments(text)

        # Process blocks
        result, extra_neg = self.process_blocks(text)

        # Clean up any remaining block markers
        result = self.block_pattern.sub('', result)

        # Clean up any empty lines and single-character lines
        result = self._cleanup_lines(result)

        return result.strip(), extra_neg.strip()

    def _find_matching_variant_close(self, text: str, start: int) -> int:
        depth = 1
        i = start
        while i < len(text) - 1:
            pair = text[i:i+2]
            if pair == "<<":
                depth += 1
                i += 2
                continue
            if pair == ">>":
                depth -= 1
                if depth == 0:
                    return i
                i += 2
                continue
            i += 1
        return -1

    def _split_variant_segments(self, content: str) -> List[str]:
        segments = []
        current = []
        angle_depth = 0
        brace_depth = 0
        bracket_depth = 0
        i = 0
        while i < len(content):
            pair = content[i:i+2]
            if pair == "<<":
                angle_depth += 1
                current.append(pair)
                i += 2
                continue
            if pair == ">>":
                if angle_depth > 0:
                    angle_depth -= 1
                current.append(pair)
                i += 2
                continue
            if content[i] == "{":
                brace_depth += 1
            elif content[i] == "}":
                brace_depth = max(0, brace_depth - 1)
            elif content[i] == "[":
                bracket_depth += 1
            elif content[i] == "]":
                bracket_depth = max(0, bracket_depth - 1)

            if pair == "||" and angle_depth == 0 and brace_depth == 0 and bracket_depth == 0:
                segments.append("".join(current))
                current = []
                i += 2
                continue

            current.append(content[i])
            i += 1

        if current:
            segments.append("".join(current))

        return segments

    def _render_variant_block(self, content: str, variant: str, allowed_variants: List[str]) -> str:
        pieces = []
        for segment in self._split_variant_segments(content):
            raw = segment
            stripped = raw.lstrip()
            label = None
            body = raw
            if len(stripped) >= 2 and stripped[1] == ":":
                candidate = stripped[0]
                if candidate in self._variant_letters:
                    label = candidate
                    body = stripped[2:].lstrip()
                else:
                    self._logger.warning("Ignoring invalid variant label '%s:' in template block.", candidate)
                    continue

            if label and label not in allowed_variants:
                self._logger.warning("Ignoring variant label '%s:' not enabled by output count.", label)
                continue

            if label is None or label == variant:
                pieces.append(body)

        combined = "".join(pieces)
        if "<<" in combined:
            return self.expand_variant_blocks(combined, variant, allowed_variants)
        return combined

    def expand_variant_blocks(self, text: str, variant: str, allowed_variants: List[str]) -> str:
        if not text:
            return text
        out = []
        i = 0
        while i < len(text):
            if text[i:i+2] == "<<":
                end = self._find_matching_variant_close(text, i + 2)
                if end == -1:
                    out.append(text[i])
                    i += 1
                    continue
                block = text[i+2:end]
                out.append(self._render_variant_block(block, variant, allowed_variants))
                i = end + 2
                continue
            out.append(text[i])
            i += 1
        return "".join(out)

    def parse_multi(self, text: str, variants: List[str]) -> Dict[str, Tuple[str, str]]:
        """Parse text into multiple variant outputs using << >> blocks."""
        resolved, extra_neg = self._parse_base(text)
        allowed = [v for v in variants if v in self._variant_letters]
        outputs: Dict[str, Tuple[str, str]] = {}
        for variant in allowed:
            pos_text, neg_text = self.expand_variant_blocks_with_neg(resolved, variant, allowed)
            pos_text = self._cleanup_lines(pos_text)
            neg_text = neg_text.strip()
            negative = ", ".join([part for part in [extra_neg, neg_text] if part])
            outputs[variant] = (pos_text.strip(), negative.strip())
        return outputs

    def expand_variant_blocks_with_neg(self, text: str, variant: str, allowed_variants: List[str]) -> Tuple[str, str]:
        if not text:
            return "", ""
        pos_parts = []
        neg_parts = []
        i = 0
        negative_mode = False
        while i < len(text):
            if text[i:i+2] == "<<":
                end = self._find_matching_variant_close(text, i + 2)
                if end == -1:
                    (neg_parts if negative_mode else pos_parts).append(text[i])
                    i += 1
                    continue
                block = text[i+2:end]
                block_pos, block_neg = self._render_variant_block_with_neg(block, variant, allowed_variants)
                if negative_mode:
                    neg_parts.append(block_pos)
                    neg_parts.append(block_neg)
                else:
                    pos_parts.append(block_pos)
                    neg_parts.append(block_neg)
                i = end + 2
                continue
            if text[i:i+2] == "!>":
                negative_mode = True
                i += 2
                continue
            target = neg_parts if negative_mode else pos_parts
            target.append(text[i])
            i += 1
        return "".join(pos_parts), "".join(neg_parts)

    def _render_variant_block_with_neg(self, content: str, variant: str, allowed_variants: List[str]) -> Tuple[str, str]:
        pos_parts = []
        neg_parts = []
        for segment in self._split_variant_segments(content):
            raw = segment
            stripped = raw.lstrip()
            label = None
            body = raw
            if len(stripped) >= 2 and stripped[1] == ":":
                candidate = stripped[0]
                if candidate in self._variant_letters:
                    label = candidate
                    body = stripped[2:].lstrip()
                else:
                    self._logger.warning("Ignoring invalid variant label '%s:' in template block.", candidate)
                    continue

            if label and label not in allowed_variants:
                self._logger.warning("Ignoring variant label '%s:' not enabled by output count.", label)
                continue

            if label is None or label == variant:
                seg_pos, seg_neg = self.expand_variant_blocks_with_neg(body, variant, allowed_variants)
                pos_parts.append(seg_pos)
                neg_parts.append(seg_neg)

        return "".join(pos_parts), "".join(neg_parts)

    def parse_option_header(self, header: str) -> Tuple[str, float]:
        """Parse option header to extract name and weight."""
        if not header:
            return "", 1.0  # Default weight when no header
            
        match = self.header_pattern.match(header)
        if not match:
            # If it's just text without a header, treat it as content with default weight
            return "", 1.0
            
        # Case 1: [name:weight]
        if match.group(1):
            name = match.group(1).strip()
            weight = match.group(2)
            if weight:
                try:
                    weight_val = float(weight)
                    if weight_val < 0:
                        raise ValueError("Weight cannot be negative")
                    return name, weight_val
                except ValueError as e:
                    if "negative" in str(e):
                        raise
                    return name, 1.0
            return name, 1.0
            
        # Case 2: [weight]
        if match.group(3):
            try:
                weight_val = float(match.group(3))
                if weight_val < 0:
                    raise ValueError("Weight cannot be negative")
                return "", weight_val
            except ValueError as e:
                if "negative" in str(e):
                    raise
                return "", 1.0
                
        return "", 1.0

    def parse_block(self, content: str) -> List[Option]:
        """Parse a block's content into a list of options."""
        options = []
        
        # Remove comments (inline-aware)
        content = self._strip_comments(content)
        
        # Split by | but preserve nested blocks and variant blocks (<< >>)
        parts = []
        current_part = []
        bracket_count = 0
        variant_depth = 0
        
        i = 0
        while i < len(content):
            pair = content[i:i+2]
            if pair == "<<":
                variant_depth += 1
                current_part.append(pair)
                i += 2
                continue
            if pair == ">>":
                if variant_depth > 0:
                    variant_depth -= 1
                current_part.append(pair)
                i += 2
                continue

            char = content[i]
            if char == '{':
                bracket_count += 1
                current_part.append(char)
            elif char == '}':
                bracket_count -= 1
                current_part.append(char)
            elif char == '|' and bracket_count == 0 and variant_depth == 0:
                parts.append(''.join(current_part).strip())
                current_part = []
            else:
                current_part.append(char)
            i += 1
        
        if current_part:
            parts.append(''.join(current_part).strip())
        
        # Parse each part
        for part in parts:
            if not part.strip():
                continue
                
            # Extract header if present
            header_match = self.header_pattern.match(part)
            if header_match:
                header = header_match.group(0)
                content = part[len(header):].strip()
            else:
                header = ""
                content = part.strip()
                
            # Parse name and weight from header
            name, weight = self.parse_option_header(header)
            
            # Handle multiline content
            content_lines = [line.strip() for line in content.split('\n') if line.strip()]
            content = '\n'.join(content_lines)
            
            # Split content and inverse content
            pos_content, neg_content = self.split_prompt(content)
            
            options.append(Option(name, weight, pos_content, neg_content))
        
        return options

    def select_option(self, options: List[Option], seed_offset: int = 0) -> Optional[Option]:
        """Select an option from the list based on weights."""
        if not options:
            return None
        
        # Validate weights are non-negative
        if any(opt.weight < 0 for opt in options):
            raise ValueError("Weights cannot be negative")
        
        # Filter out options with zero weight
        valid_options = [opt for opt in options if opt.weight > 0]
        if not valid_options:
            return None
        
        seed_value = self._seed
        if seed_value is None:
            seed_value = GlobalState.get_seed()

        if seed_value is not None:
            rng = random.Random(int(seed_value) + seed_offset)
        else:
            rng = random.Random(42 + seed_offset)

        selected = rng.choices(valid_options, weights=[opt.weight for opt in valid_options], k=1)[0]

        return selected

    def process_blocks(self, text: str) -> Tuple[str, str]:
        """Process all blocks in the text recursively."""
        result = text
        extra_neg = []
        block_count = 0  # Counter for block processing order
        
        def find_matching_brace(text: str, start: int) -> int:
            """Find the matching closing brace for an opening brace."""
            count = 1
            i = start + 1
            while i < len(text):
                if text[i] == '{':
                    count += 1
                elif text[i] == '}':
                    count -= 1
                    if count == 0:
                        return i
                i += 1
            return -1
        
        def process_block(block_content: str, depth: int = 0) -> Tuple[str, str]:
            """Process a single block's content."""
            nonlocal block_count
            block_count += 1
            
            # Parse options
            options = self.parse_block(block_content)
            if not options:
                return "", ""
            
            # Select an option using the block count as seed offset
            selected = self.select_option(options, block_count)
            if not selected:
                return "", ""
            
            # Process nested blocks in the selected content
            content = selected.content
            i = 0
            while i < len(content):
                if content[i] == '{':
                    end = find_matching_brace(content, i)
                    if end != -1:
                        nested_content = content[i+1:end]
                        nested_result, nested_neg = process_block(nested_content, depth + 1)
                        if nested_result:
                            content = content[:i] + nested_result + content[end+1:]
                            i += len(nested_result)
                        else:
                            content = content[:i] + content[end+1:]
                        if nested_neg:
                            extra_neg.append(nested_neg)
                    else:
                        i += 1
                else:
                    i += 1
            
            # Clean up the content
            content_lines = []
            for line in content.split('\n'):
                line = line.strip()
                if line and line != '.':
                    # Remove any remaining block markers and option markers
                    line = line.replace('{', '').replace('}', '')
                    if not line.startswith('[') and not line.startswith('|'):
                        content_lines.append(line)
            
            return '\n'.join(content_lines).strip(), selected.inverse_content
        
        # Process all blocks from innermost to outermost
        i = 0
        while i < len(result):
            if result[i] == '{':
                end = find_matching_brace(result, i)
                if end != -1:
                    block_content = result[i+1:end]
                    processed_content, inverse_content = process_block(block_content)
                    if processed_content:
                        result = result[:i] + processed_content + result[end+1:]
                        i += len(processed_content)
                    else:
                        result = result[:i] + result[end+1:]
                    if inverse_content:
                        extra_neg.append(inverse_content)
                else:
                    i += 1
            else:
                i += 1
        
        # Final cleanup
        lines = []
        for line in result.split('\n'):
            line = line.strip()
            if line and line != '.':
                # Remove any remaining block markers and option markers
                line = line.replace('{', '').replace('}', '')
                if not line.startswith('[') and not line.startswith('|'):
                    lines.append(line)
        
        return '\n'.join(lines), ", ".join(extra_neg)

    def parse(self, text: str, is_negative: bool = False) -> Tuple[str, str]:
        """Parse text and return the selected prompt and its inverse."""
        try:
            result, extra_neg = self._parse_base(text)
            return result.strip(), extra_neg.strip()
        except ParseError:
            raise
        except Exception as e:
            # Find the problematic part of the text
            error_pos = getattr(e, 'pos', None)
            if error_pos is not None:
                line_start = text.rfind('\n', 0, error_pos) + 1
                line_end = text.find('\n', error_pos)
                if line_end == -1:
                    line_end = len(text)
                problematic_line = text[line_start:line_end]
                line_number = text[:line_start].count('\n') + 1
                raise ParseError(
                    f"Unexpected error: {str(e)}",
                    line_number,
                    problematic_line,
                    "Please check the syntax"
                )
            raise ParseError(
                f"Unexpected error: {str(e)}",
                1,
                text.split('\n')[0],
                "Please check the syntax"
            )

def run_autotests():
    """Run automated tests to verify prompt parser behavior."""
    import sys
    from collections import Counter
    import math
    
    def debug_print(*args, **kwargs):
        print(*args, **kwargs)
        sys.stdout.flush()

    def chi_square_test(observed, expected, significance_level=0.05):
        """
        Perform chi-square test to check if observed frequencies match expected probabilities.
        Returns True if distributions are similar enough (null hypothesis not rejected).
        """
        chi_square = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
        degrees_of_freedom = len(observed) - 1
        # Critical values for significance level 0.05:
        # df=1: 3.841, df=2: 5.991, df=3: 7.815, df=4: 9.488
        critical_values = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488}
        critical_value = critical_values.get(degrees_of_freedom, 9.488)
        return chi_square <= critical_value

    debug_print("Starting autotest function...")
    parser = PromptParser()
    tests = [
        {
            "name": "Basic character selection",
            "input": """1girl
slim legs. 
{[Anne Takemaki:5]
(Ann Takamaki), platinum blonde hair
{[jewelry:1] hoop earrings | [nothing:1] .}
| [Futaba Sakura:2]
(Futaba Sakura), orange hair
{[jewelry:1] cross earrings}
}""",
            "expected": [
                "1girl",
                "slim legs.",
                "(Ann Takamaki), platinum blonde hair",
                "hoop earrings"
            ]
        },
        {
            "name": "Simple options with weights",
            "input": """1girl
{[5] red hair | [2] blue hair | [1] green hair}""",
            "expected": [
                "1girl",
                "red hair"
            ]
        },
        {
            "name": "Options with names and weights",
            "input": """1girl
{[Anna:5] red hair | [Maria:2] blue hair | [Lisa:1] green hair}""",
            "expected": [
                "1girl",
                "red hair"
            ]
        },
        {
            "name": "Options with negative prompts",
            "input": """1girl
{[5] red hair / blue hair | [2] blue hair / red hair | [1] green hair / purple hair}""",
            "expected": [
                "1girl",
                "red hair"
            ]
        },
        {
            "name": "Nested blocks",
            "input": """1girl
{[char1:2]
{[opt1:3] option1 | [opt2:1] option2}
| [char2:1]
{[opt1:1] option3 | [opt2:3] option4}
}""",
            "expected": [
                "1girl",
                "option1"
            ]
        },
        {
            "name": "Multiline options",
            "input": """1girl
{[5] red hair
long hair
beautiful eyes
| [2] blue hair
short hair
| [1] green hair}""",
            "expected": [
                "1girl",
                "red hair",
                "long hair",
                "beautiful eyes"
            ]
        },
        {
            "name": "Omitted weights",
            "input": """1girl
{red hair | [2] blue hair | green hair}""",
            "expected": [
                "1girl",
                "red hair"
            ]
        },
        {
            "name": "Zero weights",
            "input": """1girl
{[0] red hair | [2] blue hair | [1] green hair}""",
            "expected": [
                "1girl",
                "red hair"
            ]
        },
        {
            "name": "Decimal weights",
            "input": """1girl
{[0.5] red hair | [0.2] blue hair | [0.3] green hair}""",
            "expected": [
                "1girl",
                "red hair"
            ]
        },
        {
            "name": "Large weights",
            "input": """1girl
{[1000] red hair | [200] blue hair | [100] green hair}""",
            "expected": [
                "1girl",
                "red hair"
            ]
        },
        {
            "name": "Distribution test",
            "input": """1girl
{[5] red hair | [3] blue hair | [2] green hair}""",
            "expected": [
                "1girl",
                "red hair"
            ],
            "distribution_test": {
                "iterations": 1000,
                "weights": [5, 3, 2],
                "options": ["red hair", "blue hair", "green hair"]
            }
        }
    ]
    
    debug_print("\nRunning Prompt Parser Autotests...")
    all_passed = True
    
    # Set a fixed seed for reproducible tests
    debug_print("Setting random seed to 42...")
    random.seed(42)
    
    for test in tests:
        debug_print(f"\nTest: {test['name']}")
        debug_print("Input:")
        debug_print("---")
        debug_print(test['input'])
        debug_print("---")
        try:
            debug_print("Parsing input...")
            
            # Special handling for distribution test
            if "distribution_test" in test:
                dist_test = test["distribution_test"]
                iterations = dist_test["iterations"]
                weights = dist_test["weights"]
                options = dist_test["options"]
                total_weight = sum(weights)
                expected_probs = [w/total_weight for w in weights]
                
                # Run multiple iterations
                results = []
                for _ in range(iterations):
                    result, _ = parser.parse(test['input'])
                    result_lines = [line.strip() for line in result.split('\n') if line.strip()]
                    # Find which option was selected
                    for opt in options:
                        if opt in result_lines:
                            results.append(opt)
                            break
                
                # Calculate observed frequencies
                counts = Counter(results)
                observed_freqs = [counts[opt] for opt in options]
                expected_freqs = [p * iterations for p in expected_probs]
                
                # Compare distributions using chi-square test
                distributions_match = chi_square_test(observed_freqs, expected_freqs)
                
                if distributions_match:
                    debug_print("✅ PASSED - Distribution test")
                    debug_print("Expected probabilities:", [f"{p:.3f}" for p in expected_probs])
                    debug_print("Observed probabilities:", [f"{c/iterations:.3f}" for c in observed_freqs])
                else:
                    debug_print("❌ FAILED - Distribution test")
                    debug_print("Expected probabilities:", [f"{p:.3f}" for p in expected_probs])
                    debug_print("Observed probabilities:", [f"{c/iterations:.3f}" for c in observed_freqs])
                    all_passed = False
                continue
            
            # Normal test case handling
            result, _ = parser.parse(test['input'])
            debug_print("Parsing complete. Result:")
            debug_print(result)
            result_lines = [line.strip() for line in result.split('\n') if line.strip()]
            expected_lines = [line.strip() for line in test['expected']]
            
            debug_print("\nChecking results...")
            debug_print("Result lines:", result_lines)
            debug_print("Expected lines:", expected_lines)
            
            # Check if all expected lines are in the result
            missing_lines = [line for line in expected_lines if line not in result_lines]
            extra_lines = [line for line in result_lines if line not in expected_lines]
            
            if missing_lines or extra_lines:
                debug_print("❌ FAILED")
                if missing_lines:
                    debug_print("Missing lines:", missing_lines)
                if extra_lines:
                    debug_print("Extra lines:", extra_lines)
                debug_print("Result:")
                debug_print("---")
                debug_print(result)
                debug_print("---")
                debug_print("Expected:")
                debug_print("---")
                debug_print("\n".join(expected_lines))
                debug_print("---")
                all_passed = False
            else:
                debug_print("✅ PASSED")
                debug_print("Result:")
                debug_print("---")
                debug_print(result)
                debug_print("---")
        except Exception as e:
            debug_print("❌ FAILED")
            debug_print("Error:", str(e))
            debug_print("Stack trace:")
            import traceback
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            all_passed = False
    
    # Test negative weights
    debug_print("\nTesting negative weights...")
    try:
        parser.parse("""1girl
{[red hair | [-1] blue hair | green hair]}""")
        debug_print("❌ FAILED - Negative weight should raise error")
        all_passed = False
    except ValueError as e:
        if "Weight cannot be negative" in str(e):
            debug_print("✅ PASSED - Negative weight correctly rejected")
        else:
            debug_print("❌ FAILED - Wrong error message for negative weight")
            all_passed = False
    except Exception as e:
        debug_print("❌ FAILED - Wrong error type for negative weight")
        all_passed = False
    
    if all_passed:
        debug_print("\nAll tests passed successfully!")
    else:
        debug_print("\nSome tests failed. Please check the output above.")
    
    return all_passed

# Run autotests when the module is executed directly
if __name__ == "__main__":
    import sys
    print("Starting tests...", flush=True)
    try:
        run_autotests()
    except Exception as e:
        print("Error running tests:", str(e), flush=True)
        print("Stack trace:", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
    print("Tests completed.", flush=True) 
