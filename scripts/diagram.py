import requests
import urllib.parse

# Define the Mermaid diagram code
mermaid_code = """
graph TD
    A[User Query] --> B[Natural Language Understanding]
    B --> C[AI Orchestration Layer]
    C --> D[Regulatory Analyzer Agent]
    C --> E[Documentation Agent]
    C --> F[Tax Analysis Agent]
    C --> G[Incentive Identifier Agent]
    C --> H[Market Insights Agent]
    C --> I[Localization Agent]
    C --> J[Logistics Agent]
    C --> K[Currency Risk Agent]
    C --> L[Customer Support Agent]
    C --> M[Data Privacy Agent]
    C --> N[Feedback Analysis Agent]
    D --> O[Knowledge Management Layer]
    E --> O
    F --> O
    G --> O
    H --> O
    I --> O
    J --> O
    K --> O
    L --> O
    M --> O
    N --> O
    O --> P[Validation Layer]
    P --> Q[Generate Response]
    Q --> R[Deliver to User]
"""

# Encode the Mermaid code for URL
encoded_mermaid = urllib.parse.quote(mermaid_code)

# Define the Mermaid Live Editor API endpoint
api_url = f"https://mermaid.ink/img/{encoded_mermaid}"

# Make a GET request to fetch the SVG image
response = requests.get(api_url)

if response.status_code == 200:
    # Save the SVG image to a file
    with open("workflow_diagram.svg", "wb") as file:
        file.write(response.content)
    print("Workflow diagram downloaded successfully as 'workflow_diagram.svg'.")
else:
    print(f"Failed to download workflow diagram. Status code: {response.status_code}")