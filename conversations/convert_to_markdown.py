#!/usr/bin/env python3
"""
Convert conversation.py files (Anthropic API message format) to Markdown.
Removes API keys and signatures while preserving the conversation.
"""

import ast
import sys
import re
from pathlib import Path


def extract_messages_from_file(file_path):
    """Extract the messages list from a Python file containing Anthropic API calls."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Parse the Python file
    tree = ast.parse(content)

    # Find the messages list in the AST
    messages = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Look for client.messages.create calls
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == 'create':
                    # Extract the messages keyword argument
                    for keyword in node.keywords:
                        if keyword.arg == 'messages':
                            messages = ast.literal_eval(ast.unparse(keyword.value))
                            break
            if messages:
                break

    return messages


def format_message_content(content):
    """Extract text from content list."""
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                text_parts.append(item.get('text', ''))
        return '\n'.join(text_parts)
    return str(content)


def remove_signatures(text):
    """Remove common signatures and sign-offs from messages."""
    # Common signature patterns - being conservative to avoid removing markdown
    # Note: Removed the "---" pattern as it conflicts with markdown horizontal rules
    signature_patterns = [
        r'\n\nBest regards,.*$',
        r'\n\nBest,.*$',
        r'\n\nSincerely,.*$',
        r'\n\nThanks,.*$',
        r'\n\nRegards,.*$',
        r'\n\nCheers,.*$',
        r'\n\nWarm regards,.*$',
        r'\n\n- [A-Z]\w+\s*$',  # - Name at the end (capitalized name)
    ]

    result = text
    for pattern in signature_patterns:
        result = re.sub(pattern, '', result, flags=re.DOTALL | re.MULTILINE)

    return result.strip()


def messages_to_markdown(messages, remove_sigs=True):
    """Convert messages list to Markdown format."""
    md_parts = []

    for i, msg in enumerate(messages):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')

        # Format the content
        text = format_message_content(content)

        # Remove signatures if requested
        if remove_sigs:
            text = remove_signatures(text)

        # Create markdown section
        if role == 'user':
            md_parts.append(f"## User\n\n{text}\n")
        elif role == 'assistant':
            md_parts.append(f"## Assistant\n\n{text}\n")
        else:
            md_parts.append(f"## {role.title()}\n\n{text}\n")

    return '\n'.join(md_parts)


def convert_file(input_path, output_path=None, remove_sigs=True):
    """Convert a conversation.py file to Markdown."""
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_suffix('.md')
    else:
        output_path = Path(output_path)

    print(f"Reading {input_path}...")
    messages = extract_messages_from_file(input_path)

    if not messages:
        print("Error: Could not find messages in the file.")
        sys.exit(1)

    print(f"Found {len(messages)} messages")

    print(f"Converting to Markdown...")
    markdown = messages_to_markdown(messages, remove_sigs=remove_sigs)

    # Add header
    header = f"# Conversation\n\n*Converted from {input_path.name}*\n\n---\n\n"
    markdown = header + markdown

    print(f"Writing to {output_path}...")
    with open(output_path, 'w') as f:
        f.write(markdown)

    print(f"Done! Conversation saved to {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_to_markdown.py <input_file.py> [output_file.md]")
        print("Example: python convert_to_markdown.py conversation.py")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    convert_file(input_file, output_file, remove_sigs=True)


if __name__ == "__main__":
    main()
