---
name: Bug Report
about: Report a bug or unexpected behavior
title: "[BUG] "
labels: bug
assignees: ''

body:
  - type: textarea
    id: description
    attributes:
      label: Description
      description: A clear and concise description of the bug.
    validations:
      required: true
  - type: textarea
    id: steps
    attributes:
      label: Steps to Reproduce
      description: Steps to reproduce the behavior.
      placeholder: |
        1. Go to '...'
        2. Click on '....'
        3. Scroll down to '....'
        4. See error
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: Expected Behavior
      description: What you expected to happen.
    validations:
      required: true
  - type: textarea
    id: environment
    attributes:
      label: Environment
      description: OS, Rust version, etc.
      placeholder: |
        - OS: [e.g., Ubuntu 22.04]
        - Rust: [e.g., 1.70]
        - Python: [e.g., 3.11]
    validations:
      required: true
  - type: textarea
    id: additional
    attributes:
      label: Additional Context
      description: Any other context or screenshots.