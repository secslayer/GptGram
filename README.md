# GptGram 

## ✨ Introduction

GptGram tackles the challenge of fragmented AI knowledge by enabling intelligent, secure, and collaborative interactions between specialized AI agents. This document focuses on the core algorithms that drive this collaboration and guides you through setting up the system locally, entirely powered by Ollama models.

## 🧠 Core Algorithms in Detail

### 1. Semantic Embedding: Representing Agent Expertise

Before agents can collaborate intelligently, GptGram first needs to understand what each agent is good at. This is achieved through **semantic embedding**.

* **Process:** When an agent is created or its profile (description, specialization) is updated, its textual information is fed into a Large Language Model (LLM) capable of generating embeddings (e.g., a text embedding model).

* **Output:** The LLM transforms the agent's textual description (e.g., "Financial Analyst, specializing in market trends and investment advice") into a high-dimensional numerical vector (an embedding). This vector captures the semantic meaning of the agent's expertise, placing it in a "vector space" where similar concepts are numerically closer.

* **LLM Role:** **Ollama** (for local embedding models like `nomic-embed-text`) is used for this purpose.

### 2. Graph Neural Network (GNN): Contextualizing Agent Relationships

While raw semantic embeddings provide a good starting point, GNNs take it a step further by understanding agents *in the context of their connections*.

* **The Graph:** GptGram models the network of agents as a graph:

    * **Nodes:** Each AI agent is a node in the graph. The initial feature vector (`x`) for each node is its semantic embedding (as described above).

    * **Edges:** Connections (`edge_index`) between agents are formed based on historical collaborations or pre-defined relationships. For example, if "Agent A" collaborated with "Agent B" on a query, an edge exists.

* **GNN Processing:** The `AgentGraphNN` (a GCNConv-based model in your backend implementation) processes this graph data:

    1.  **Message Passing:** Each node (agent) iteratively aggregates information from its direct neighbors (agents it's connected to).

    2.  **Feature Transformation:** The GNN layers (`GCNConv`) transform the node features by combining the agent's own embedding with the aggregated information from its neighbors.

    3.  **Output:** The GNN produces a new set of "GNN-embeddings" for each agent. These embeddings are richer than the initial semantic embeddings because they incorporate information about the agent's position within the network and the expertise of its collaborators. An agent's GNN-embedding will reflect not just *what it knows*, but *who it knows* and *who it has successfully collaborated with*.

* **Benefit:** This contextual understanding allows GptGram to find more relevant collaborators. Two agents might have similar individual expertise, but if the GNN shows they are also frequently linked in collaborative problem-solving contexts (even if indirectly), their GNN-embeddings will be closer, indicating a higher potential for effective future collaboration.

    * **Relevant Research:**

        * **Graph Convolutional Networks (GCNs):** A foundational GNN model for semi-supervised classification on graph-structured data.

            * **Paper:** Thomas N. Kipf, Max Welling. "Semi-Supervised Classification with Graph Convolutional Networks." (ICLR 2017) \[[arXiv:1609.02907](https://arxiv.org/abs/1609.02907)\]

        * **Message Passing Neural Networks (MPNNs):** A general framework that encompasses many GNN variants.

            * **Paper:** Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl. "Neural Message Passing for Quantum Chemistry." (ICML 2017) \[[arXiv:1704.01212](https://arxiv.org/abs/1704.01212)\]

### 3. Dynamic Collaboration Orchestration & Knowledge Synthesis

This is where the GNN's output is actively leveraged.

* **Confidence Threshold:** When a primary agent processes a query, it first generates an initial response and calculates its confidence based on the semantic similarity between the query and its own response. If this confidence falls below a pre-defined `collaboration_threshold`, the agent initiates a search for help.

* **GNN-driven Discovery:**

    1.  The primary agent queries the platform to get its own GNN-embedding and the GNN-embeddings of all other active agents.

    2.  It then calculates the **cosine similarity** between its own GNN-embedding and the GNN-embeddings of other agents. Cosine similarity measures the angle between two vectors, indicating how directionally similar they are (i.e., how semantically related their expertise and network context are).

    3.  Agents with a similarity score above a `min_similarity_for_collaboration` threshold are identified as potential collaborators. The system prioritizes the most similar agents.

* **Orchestration:** The primary agent sends the original query to the selected highly similar collaborating agents.

* **Synthesis:** Each collaborating agent provides its own response. The primary agent then collects these responses and uses a **synthesis prompt** to instruct its LLM (using its own `llm_interface`, which is Ollama in this setup) to combine these diverse perspectives into a single, comprehensive, and high-confidence answer. This final synthesized response is then presented to the user.

* **Learning & Feedback:** Successful collaborations (and their resulting synthesized knowledge) can implicitly strengthen the "edges" in the underlying graph, further influencing future GNN-embeddings and improving future agent discovery. This forms a continuous learning loop.

## 🚀 Getting Started

Follow these steps to get your GptGram Agentic Network up and running locally.

### Prerequisites

* **Python 3.9+**

* **`pip`** (Python package installer)

* **Ollama:** For running local LLMs. Download and install Ollama from [ollama.com](https://ollama.com/). After installation, pull a text generation model (e.g., `ollama pull llama2`) and an embedding model (e.g., `ollama pull nomic-embed-text`). Ensure `ollama serve` is running.

### Installation

1.  **Clone the Repository:**

    ```
    git clone [https://github.com/your-username/gptgram.git](https://github.com/your-username/gptgram.git) # Replace with your actual repo URL
    cd gptgram
    
    ```

2.  **Install Dependencies:**
    Create a `requirements.txt` file in the root of your project with the following content:

    ```
    Flask
    requests
    numpy
    scikit-learn
    torch
    torch-geometric
    networkx
    
    ```

    Then, install them:

    ```
    pip install -r requirements.txt
    
    ```

3.  **Database Setup:**
    The `init_db()` function in `app.py` will automatically create the `agent_graph.db` file and necessary tables if they don't exist when the application starts.

### Configuration

* **Ollama Host (if not default):**
    If your Ollama server is not running on `http://localhost:11434`, you can set the `OLLAMA_HOST` environment variable. For example:

    ```
    export OLLAMA_HOST="http://your_ollama_ip:11434"
    
    ```

* **(Optional) Ollama Timeouts/Retries:**

    ```
    export OLLAMA_TIMEOUT="60"
    export OLLAMA_MAX_RETRIES="3"
    export OLLAMA_RETRY_DELAY="1"
    
    ```

### Running the Application

With all prerequisites installed and configured, navigate to your project's root directory in the terminal and start the Flask backend:
```
python app.py
```
The application will typically run on `http://127.0.0.1:5001`. Open this URL in your web browser to access the GptGram web interface (`index.html`) for agent management and interaction. The system will also proactively check if your Ollama server is running.
