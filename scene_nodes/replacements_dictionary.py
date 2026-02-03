from typing import Dict, Optional

class ReplaceDict:
    """
    A node that allows defining a single key-value pair for text replacement.
    Can merge with an optional input dictionary.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": ("STRING", {"default": "", "multiline": False}),
                "value": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "input_dict": ("DICT",),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("replacements",)
    FUNCTION = "create_replacements"
    CATEGORY = "ESS/PromptBuilder"

    def create_replacements(self, key: str, value: str, input_dict: Optional[Dict[str, str]] = None) -> tuple:
        """
        Create a dictionary of replacements from the input key-value pair and optional input dictionary.
        
        Args:
            key: The key to replace
            value: The replacement value
            input_dict: Optional input dictionary to merge with
            
        Returns:
            Tuple containing the merged replacements dictionary
        """
        # Start with input dictionary if provided, otherwise empty dict
        replacements = input_dict.copy() if input_dict else {}
        
        # Only add the key-value pair if both are non-empty
        if key.strip() and value.strip():
            replacements[key.strip()] = value.strip()
                
        return (replacements,) 