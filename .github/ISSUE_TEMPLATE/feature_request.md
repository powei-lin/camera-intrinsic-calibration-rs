---
name: Feature Request
about: Suggest a new feature or enhancement
title: "[FEATURE] "
labels: enhancement
assignees: ''

body:
  - type: textarea
    id: summary
    attributes:
      label: Summary
      description: A brief summary of the feature.
    validations:
      required: true
  - type: textarea
    id: problem
    attributes:
      label: Problem
      description: What problem does this solve?
    validations:
      required: true
  - type: textarea
    id: solution
    attributes:
      label: Proposed Solution
      description: Describe your proposed solution.
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives Considered
      description: Any alternative solutions?
  - type: textarea
    id: additional
    attributes:
      label: Additional Context
      description: Any other context or details.