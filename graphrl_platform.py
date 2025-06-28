import sqlite3
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import requests
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime
import threading
import time
import uuid
from typing import Dict, List, Any, Optional
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# export OLLAMA_HOST="http://localhost:11434"
# export OLLAMA_TIMEOUT="60"       
# export OLLAMA_MAX_RETRIES="5"     
# export OLLAMA_RETRY_DELAY="2"     
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
OLLAMA_INITIAL_RETRY_DELAY = int(os.getenv("OLLAMA_RETRY_DELAY", "1"))


def convert_non_serializable_types(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.ndarray, list, tuple)):
        return type(obj)(convert_non_serializable_types(elem) for elem in obj)
    elif isinstance(obj, dict):
        return {k: convert_non_serializable_types(v) for k, v in obj.items()}
    else:
        return obj


def init_db():
    conn = sqlite3.connect('agent_graph.db', check_same_thread=False)
    cursor = conn.cursor()

  
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            specialization TEXT,
            model_config TEXT,
            knowledge_base TEXT,
            created_at TIMESTAMP,
            user_id TEXT,
            status TEXT DEFAULT 'active',
            system_prompt_template TEXT
        )
    ''')

   
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS agent_connections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_agent TEXT,
            target_agent TEXT,
            interaction_type TEXT,
            knowledge_shared TEXT,
            similarity_score REAL,
            created_at TIMESTAMP,
            FOREIGN KEY (source_agent) REFERENCES agents (id),
            FOREIGN KEY (target_agent) REFERENCES agents (id)
        )
    ''')

   
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS queries (
            id TEXT PRIMARY KEY,
            agent_id TEXT,
            query TEXT,
            response TEXT,
            embedding TEXT,
            solved BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP,
            FOREIGN KEY (agent_id) REFERENCES agents (id)
        )
    ''')

    conn.commit()
    return conn


class AgentGraphNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AgentGraphNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        
        if batch is not None:
            x = global_mean_pool(x, batch) 
        return x


GNN_EMBEDDING_DIM = 768
AGENT_GNN_MODEL: Optional[AgentGraphNN] = None
GLOBAL_OLLAMA_INTERFACE = None 


def _check_ollama_status_internal(host: str, timeout: int) -> bool:

    try:
        response = requests.get(f"{host}/", timeout=min(5, timeout)) # Quick check
        response.raise_for_status()
        return True
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
            requests.exceptions.RequestException) as e:
        # print(f"DEBUG: Ollama status check failed: {e}") # Enable for deeper debug
        return False

def call_ollama_api(
    endpoint: str,
    method: str = "POST",
    payload: dict = None,
    host: str = OLLAMA_HOST,
    timeout: int = OLLAMA_TIMEOUT,
    max_retries: int = OLLAMA_MAX_RETRIES,
    initial_delay: int = OLLAMA_INITIAL_RETRY_DELAY
) -> dict:

    current_delay = initial_delay
    full_url = f"{host}{endpoint}"

    for attempt in range(1, max_retries + 1):
        try:
           
            if attempt == 1 and not _check_ollama_status_internal(host, timeout):
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARNING: Ollama server not reachable at {host}. Attempting to proceed with retries.")
            elif attempt > 1:
                 if not _check_ollama_status_internal(host, timeout):
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARNING: Still unable to reach Ollama at {host}. Retrying in {current_delay}s... (Attempt {attempt}/{max_retries})")
                    time.sleep(current_delay)
                    current_delay *= 2
                    continue 

            if method.upper() == "POST":
                response = requests.post(full_url, json=payload, timeout=timeout)
            elif method.upper() == "GET":
                response = requests.get(full_url, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}. Use 'GET' or 'POST'.")

            response.raise_for_status()
            return response.json()

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < max_retries:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Connection/Timeout error with Ollama ({endpoint}): {e}. Retrying in {current_delay}s... (Attempt {attempt}/{max_retries})")
                time.sleep(current_delay)
                current_delay *= 2
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CRITICAL: Failed to connect/timeout with Ollama at {host} after {max_retries} attempts.")
                raise ConnectionError(f"Failed to connect/timeout with Ollama after {max_retries} attempts.") from e
        except requests.exceptions.RequestException as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Request to Ollama ({endpoint}) failed: {e}. (Attempt {attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(current_delay) 
                current_delay *= 2
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CRITICAL: Ollama request to {endpoint} failed after {max_retries} attempts due to: {e}")
                raise RuntimeError(f"Ollama request failed after {max_retries} attempts due to: {e}") from e
    return {} 


class OllamaInterface:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    def generate(self, model="llama2", prompt="", system=""):
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False
        }
        try:
            response_json = call_ollama_api(
                endpoint="/api/generate",
                method="POST",
                payload=payload,
                host=self.base_url
            )
            return response_json.get("response", "") if response_json else ""
        except (ConnectionError, RuntimeError):
            return ""

    def embed(self, model="llama2", text=""):
        if not text:
            return []

        payload = {"model": model, "prompt": text}
        try:
            response_json = call_ollama_api(
                endpoint="/api/embeddings",
                method="POST",
                payload=payload,
                host=self.base_url
            )
            return response_json.get("embedding", []) if response_json else []
        except (ConnectionError, RuntimeError):
            return [] 


class AIAgent:
    def __init__(self, agent_id, name, description, specialization, model_config, system_prompt_template, db_conn):
        self.id = agent_id
        self.name = name
        self.description = description
        self.specialization = specialization
        self.model_config = json.loads(model_config) if isinstance(model_config, str) else model_config
        self.system_prompt_template = system_prompt_template
        self.db = db_conn
        self.ollama = GLOBAL_OLLAMA_INTERFACE 
        self.knowledge_base = []
        self.collaboration_threshold = self.model_config.get("collaboration_threshold", 0.7)
        self.min_similarity_for_collaboration = self.model_config.get("min_similarity_for_collaboration", 0.1)

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query, potentially seeking help from other agents"""
        query_id = str(uuid.uuid4())

        initial_response = self._generate_response(query)
        initial_confidence = self._calculate_confidence(query, initial_response)

        current_response = initial_response
        final_confidence = initial_confidence
        collaboration_used = False
        collaborative_responses = []

        print(f"[{self.name}] Self-generated response. Initial Confidence: {initial_confidence:.2f}")

        if initial_confidence < self.collaboration_threshold:
            print(f"[{self.name}] Initial Confidence low ({initial_confidence:.2f} < {self.collaboration_threshold}). Seeking collaboration...")
            collaboration_result = self._seek_collaboration(query)
            
            if collaboration_result:
                collaboration_used = True
                collaborative_responses = collaboration_result["collaborative_responses"]
                current_response = collaboration_result["response"]

                final_confidence = self._calculate_confidence(query, current_response)
                print(f"[{self.name}] Collaboration successful. Final Confidence (after synthesis): {final_confidence:.2f}")
            else:
                print(f"[{self.name}] No relevant agents found for collaboration or collaboration failed.")
        else:
            print(f"[{self.name}] Initial Confidence high ({initial_confidence:.2f} >= {self.collaboration_threshold}). No collaboration needed.")

        self._store_query(query_id, query, current_response)

        return {
            "query_id": query_id,
            "agent_id": self.id,
            "query": query,
            "response": current_response,
            "confidence": final_confidence,
            "collaboration_used": collaboration_used,
            "collaborative_responses": collaborative_responses
        }

    def _generate_response(self, query: str) -> str:
        """Generate response using Ollama with a configurable system prompt."""
        system_prompt = self.system_prompt_template.format(
            agent_name=self.name,
            specialization=self.specialization,
            description=self.description
        )
        return self.ollama.generate(
            model=self.model_config.get("model", "llama2"),
            prompt=query,
            system=system_prompt
        )

    def _calculate_confidence(self, query: str, response: str) -> float:

        if not response:
            print("[DEBUG Confidence] Response is empty. Returning 0.1.")
            return 0.1

        embedding_model = self.model_config.get("model", "llama2")
        query_embedding = self.ollama.embed(model=embedding_model, text=query)
        response_embedding = self.ollama.embed(model=embedding_model, text=response)

        if not query_embedding or not response_embedding:
            print(f"[DEBUG Confidence] Warning: Could not get embeddings for confidence calculation. Q_emb_empty: {not bool(query_embedding)}, R_emb_empty: {not bool(response_embedding)}. Falling back to basic heuristic.")
            return self._calculate_confidence_heuristic_fallback(query, response)

        query_vec = np.array(query_embedding)
        response_vec = np.array(response_embedding)

        norm_query = np.linalg.norm(query_vec)
        norm_response = np.linalg.norm(response_vec)

        if norm_query == 0 or norm_response == 0:
            print("[DEBUG Confidence] One or both embedding norms are zero. Returning 0.1.")
            return 0.1

        similarity = np.dot(query_vec, response_vec) / (norm_query * norm_response)
        confidence = max(0.1, (similarity + 1) / 2)
        final_confidence = min(1.0, confidence)

        print(f"[DEBUG Confidence] Calculated Confidence: {final_confidence:.4f}")
        return float(final_confidence)

    def _calculate_confidence_heuristic_fallback(self, query: str, response: str) -> float:

        print("[DEBUG Confidence] Using heuristic fallback.")
        uncertainty_keywords = ["not sure", "don't know", "uncertain", "maybe", "possibly", "might be", "apologize", "no access", "cannot provide"]
        response_lower = response.lower()
        uncertainty_count = sum(1 for keyword in uncertainty_keywords if keyword in response_lower)
        base_confidence = min(1.0, len(response) / 200.0)
        uncertainty_penalty = uncertainty_count * 0.2
        return max(0.1, base_confidence - uncertainty_penalty)


    def _get_agent_graph_data(self) -> Optional[Data]:

        cursor = self.db.cursor()
        cursor.execute("SELECT id, description, specialization FROM agents WHERE status = 'active'")
        all_agents_data = cursor.fetchall()

        if not all_agents_data:
            return None, {}

        agent_id_to_idx = {agent_id: i for i, (agent_id, _, _) in enumerate(all_agents_data)}
        num_nodes = len(all_agents_data)

        node_features_list = []
        for agent_id, description, specialization in all_agents_data:
            combined_text = f"{description or ''} {specialization or ''}".strip()
            embedding = self.ollama.embed(model=self.model_config.get("model", "llama2"), text=combined_text)
            if not embedding:
                embedding = np.zeros(GNN_EMBEDDING_DIM).tolist()
                print(f"Warning: Could not get embedding for agent {agent_id}. Using zero vector.")
            node_features_list.append(embedding)
        
        x = torch.tensor(node_features_list, dtype=torch.float)

        cursor.execute("SELECT source_agent, target_agent FROM agent_connections")
        connections = cursor.fetchall()
        
        edge_list = []
        for src_id, tgt_id in connections:
            if src_id in agent_id_to_idx and tgt_id in agent_id_to_idx:
                edge_list.append([agent_id_to_idx[src_id], agent_id_to_idx[tgt_id]])
                edge_list.append([agent_id_to_idx[tgt_id], agent_id_to_idx[src_id]]) # Add symmetric edge

        unique_edges = list(set(tuple(sorted(e)) for e in edge_list))
        
        if not unique_edges:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            final_edge_list = []
            for u, v in unique_edges:
                final_edge_list.append([u,v])
                final_edge_list.append([v,u]) # Add symmetric edge explicitly
            edge_index = torch.tensor(final_edge_list, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index), agent_id_to_idx

    def _get_gnn_agent_embeddings(self) -> Dict[str, np.ndarray]:
        global AGENT_GNN_MODEL

        graph_data, agent_id_to_idx = self._get_agent_graph_data()
        if graph_data is None or not agent_id_to_idx:
            return {}

        if AGENT_GNN_MODEL is None:
            AGENT_GNN_MODEL = AgentGraphNN(graph_data.x.shape[1], 256, GNN_EMBEDDING_DIM)
            print("Initialized AGENT_GNN_MODEL (first time).")

        AGENT_GNN_MODEL.eval()
        with torch.no_grad():
            node_embeddings = AGENT_GNN_MODEL(graph_data.x, graph_data.edge_index)
        
        agent_embeddings = {}
        idx_to_agent_id = {v: k for k, v in agent_id_to_idx.items()}
        for i, emb in enumerate(node_embeddings.cpu().numpy()):
            agent_embeddings[idx_to_agent_id[i]] = emb
        
        return agent_embeddings


    def _seek_collaboration(self, query: str) -> Optional[Dict[str, Any]]:
        cursor = self.db.cursor()
        cursor.execute("SELECT id FROM agents WHERE id != ? AND status = 'active'", (self.id,))
        other_agent_ids = [row[0] for row in cursor.fetchall()]

        if not other_agent_ids:
            return None

        all_agent_gnn_embeddings = self._get_gnn_agent_embeddings()
        
        if self.id not in all_agent_gnn_embeddings:
            print(f"Error: Current agent {self.id} not found in GNN embeddings. Cannot proceed with GNN-based collaboration.")
            return None
        
        self_embedding = all_agent_gnn_embeddings[self.id]

        agent_similarities = []
        for other_agent_id in other_agent_ids:
            if other_agent_id in all_agent_gnn_embeddings:
                other_agent_embedding = all_agent_gnn_embeddings[other_agent_id]
                
                similarity = cosine_similarity(self_embedding.reshape(1, -1), other_agent_embedding.reshape(1, -1))[0][0]
                agent_similarities.append((other_agent_id, float(similarity))) # Convert numpy float to python float
            else:
                print(f"Warning: Other agent {other_agent_id} not found in GNN embeddings.")

        similar_agents = sorted(
            [a for a in agent_similarities if a[1] > self.min_similarity_for_collaboration],
            key=lambda x: x[1], reverse=True
        )

        print(f"[{self.name}] Found similar agents using GNN embeddings: {similar_agents}")

        if not similar_agents:
            return None

        collaborative_responses = []
        collaborating_agents = []

        for agent_id, similarity in similar_agents[:3]:
            other_agent_data = self._get_agent_data_by_id(agent_id)
            if other_agent_data:
                other_agent = AIAgent(
                    other_agent_data[0], other_agent_data[1], other_agent_data[2], 
                    other_agent_data[3], other_agent_data[4], other_agent_data[5], self.db
                )
                print(f"[{self.name}] Collaborating with {other_agent.name} (Similarity: {similarity:.2f})")
                response_from_collab = other_agent._generate_response(query)
                if response_from_collab:
                    collaborative_responses.append({
                        "agent_id": agent_id,
                        "agent_name": other_agent.name,
                        "response": response_from_collab,
                        "similarity": similarity
                    })
                    collaborating_agents.append(agent_id)

                    self._record_collaboration(agent_id, query, response_from_collab, similarity)
                else:
                    print(f"[{self.name}] Warning: Collaboration with {other_agent.name} failed to generate a response (Ollama issue?). Skipping this collaborator.")
            else:
                print(f"[{self.name}] Failed to get agent instance data for ID: {agent_id}")

        if collaborative_responses:
            combined_response = self._combine_responses(query, collaborative_responses)
            return {
                "response": combined_response,
                "collaborating_agents": collaborating_agents,
                "collaborative_responses": collaborative_responses
            }

        return None

    def _combine_responses(self, original_query: str, collab_responses: List[Dict[str, Any]]) -> str:
        """Combines responses from collaborating agents using a synthesis prompt."""
        if not collab_responses:
            return "No collaborative responses to synthesize."

        combined_text = f"Original Query: {original_query}\n\n"
        for i, collab in enumerate(collab_responses):
            combined_text += f"--- Contribution from {collab['agent_name']} (Similarity {collab['similarity']:.2f}) ---\n"
            combined_text += f"{collab['response']}\n\n"
        
        synthesis_prompt = f"""You are an expert AI tasked with synthesizing a comprehensive answer from multiple agent contributions.
        Original User Query: {original_query}
        
        Collaborative Responses Provided:
        {combined_text}

        Based on the above collaborative responses, provide a single, coherent, and comprehensive answer to the original user query. If the collaborating agents collectively indicate that the information is not available or is uncertain, clearly state that in your synthesized answer. Do not invent information."""

        final_synthesis = self.ollama.generate(
            model=self.model_config.get("model", "llama2"),
            prompt=synthesis_prompt,
            system="You are a brilliant synthesizer of information, prioritizing accuracy and acknowledging limitations based on provided data."
        )
        return final_synthesis if final_synthesis else "Could not synthesize a combined response from collaboration, possibly due to an Ollama error."


    def _record_collaboration(self, target_agent: str, query: str, shared_knowledge: str, similarity: float):
        """Record collaboration between agents"""
        cursor = self.db.cursor()
        cursor.execute('''
            INSERT INTO agent_connections (source_agent, target_agent, interaction_type, knowledge_shared, similarity_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (self.id, target_agent, "collaboration", f"Query: {query[:100]}...", similarity, datetime.now()))
        self.db.commit()
        print(f"[{self.name}] Recorded collaboration with {self.get_agent_name_by_id(target_agent)}. DB committed.")


    def _store_query(self, query_id: str, query_text: str, response_text: str):
        """Store the query and its response in the database."""
        cursor = self.db.cursor()
        cursor.execute('''
            INSERT INTO queries (id, agent_id, query, response, solved, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (query_id, self.id, query_text, response_text, True, datetime.now()))
        self.db.commit()

    def _get_agent_data_by_id(self, agent_id: str):
        """Retrieve raw agent data by ID from the database."""
        cursor = self.db.cursor()
        cursor.execute("SELECT id, name, description, specialization, model_config, system_prompt_template FROM agents WHERE id = ?", (agent_id,))
        return cursor.fetchone()
    
    def get_agent_name_by_id(self, agent_id: str) -> str:
        """Helper to get agent name from ID."""
        cursor = self.db.cursor()
        cursor.execute("SELECT name FROM agents WHERE id = ?", (agent_id,))
        result = cursor.fetchone()
        return result[0] if result else "Unknown Agent"


app = Flask(__name__)
db_connection = init_db()
GLOBAL_OLLAMA_INTERFACE = OllamaInterface()

@app.route('/')
def index():
    if not os.path.exists('index.html'):
        try:
            with open("index.html", "w") as f:
                f.write(html_template)
            print("Generated index.html successfully.")
        except IOError as e:
            print(f"Error writing index.html: {e}")
            print("Please ensure you have write permissions in the current directory.")
            return render_template_string(html_template)
    return render_template_string(open('index.html').read())

@app.route('/api/create_agent', methods=['POST'])
def create_agent_route():
    data = request.json
    agent_id = str(uuid.uuid4())
    name = data.get('name')
    description = data.get('description')
    specialization = data.get('specialization')
    model_config = json.dumps(data.get('model_config', {"model": "llama2"}))
    user_id = data.get('user_id', 'default_user')
    system_prompt_template = data.get('system_prompt_template', 
        "You are {agent_name}, specialized in {specialization}. Description: {description}. Answer accurately and concisely based on your expertise. If you lack information, state it clearly."
    )

    if not name or not specialization:
        return jsonify({"error": "Name and specialization are required"}), 400

    cursor = db_connection.cursor()
    try:
        cursor.execute('''
            INSERT INTO agents (id, name, description, specialization, model_config, created_at, user_id, system_prompt_template)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (agent_id, name, description, specialization, model_config, datetime.now(), user_id, system_prompt_template))
        db_connection.commit()
        return jsonify({"message": "Agent created successfully", "agent_id": agent_id}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Agent with this ID already exists (very rare with UUIDs)"}), 409
    except Exception as e:
        db_connection.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/api/agents', methods=['GET'])
def get_agents_route():
    cursor = db_connection.cursor()
    cursor.execute("SELECT id, name, description, specialization, model_config, created_at, user_id, status, system_prompt_template FROM agents")
    agents = []
    for row in cursor.fetchall():
        agents.append({
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "specialization": row[3],
            "model_config": json.loads(row[4]),
            "created_at": row[5],
            "user_id": row[6],
            "status": row[7],
            "system_prompt_template": row[8]
        })
    return jsonify(agents)

@app.route('/api/agent/<agent_id>/query', methods=['POST'])
def query_agent_route(agent_id):
    query_text = request.json.get('query')
    if not query_text:
        return jsonify({"error": "Query text is required"}), 400

    cursor = db_connection.cursor()
    cursor.execute("SELECT id, name, description, specialization, model_config, system_prompt_template FROM agents WHERE id = ?", (agent_id,))
    agent_data = cursor.fetchone()

    if not agent_data:
        return jsonify({"error": "Agent not found"}), 404

    agent = AIAgent(
        agent_data[0], agent_data[1], agent_data[2], agent_data[3], 
        agent_data[4], agent_data[5], db_connection
    )
    result = agent.process_query(query_text)
    
    serializable_result = convert_non_serializable_types(result)
    return jsonify(serializable_result)

@app.route('/api/stats', methods=['GET'])
def get_stats_route():
    cursor = db_connection.cursor()

    cursor.execute("SELECT COUNT(*) FROM agents")
    total_agents = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM agent_connections")
    total_connections = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM queries")
    total_queries = cursor.fetchone()[0]
    
    top_source_agents = []
    cursor.execute("""
        SELECT source_agent, COUNT(*) as count
        FROM agent_connections
        GROUP BY source_agent
        ORDER BY count DESC LIMIT 5
    """)
    for agent_id, count in cursor.fetchall():
      
        agent = AIAgent(agent_id, "", "", "", {}, "", db_connection) 
        top_source_agents.append({"agent_name": agent.get_agent_name_by_id(agent_id), "count": count})

    top_target_agents = []
    cursor.execute("""
        SELECT target_agent, COUNT(*) as count
        FROM agent_connections
        GROUP BY target_agent
        ORDER BY count DESC LIMIT 5
    """)
    for agent_id, count in cursor.fetchall():
       
        agent = AIAgent(agent_id, "", "", "", {}, "", db_connection) 
        top_target_agents.append({"agent_name": agent.get_agent_name_by_id(agent_id), "count": count})


    stats_data = {
        "total_agents": total_agents,
        "total_connections": total_connections,
        "total_queries": total_queries,
        "top_source_agents": top_source_agents,
        "top_target_agents": top_target_agents
    }
    serializable_stats = convert_non_serializable_types(stats_data)
    return jsonify(serializable_stats)


html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GraphRL AI Agent Platform</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #343a40;
            line-height: 1.6;
        }
        h1, h2, h3 {
            color: #212529;
            margin-top: 0;
            margin-bottom: 15px;
        }
        .container {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        .panel {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 25px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
            display: flex;
            flex-direction: column;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #495057;
        }
        .form-group input,
        .form-group textarea,
        .form-group select {
            width: calc(100% - 20px);
            padding: 10px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 1em;
            box-sizing: border-box;
            color: #495057;
        }
        .form-group textarea {
            resize: vertical;
            min-height: 60px;
        }
        .button {
            background-color: #007bff;
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1em;
            transition: background-color 0.2s ease;
            width: 100%;
            box-sizing: border-box;
        }
        .button:hover {
            background-color: #0056b3;
        }
        .agent-list-container {
            margin-top: 25px;
            overflow-y: auto;
            max-height: 400px;
            border-top: 1px solid #eee;
            padding-top: 15px;
        }
        .agent-card {
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #fcfdfe;
        }
        .agent-card h3 {
            margin-top: 0;
            margin-bottom: 10px;
            color: #007bff;
        }
        .agent-card p {
            margin-bottom: 5px;
            font-size: 0.95em;
            color: #555;
        }
        #queryResult {
            background-color: #e9f7ef;
            border: 1px solid #d4edda;
            padding: 20px;
            border-radius: 6px;
            margin-top: 25px;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 0.95em;
        }
        #queryResult h3 { color: #28a745; }
        #queryResult h4 { margin-top: 15px; margin-bottom: 10px; color: #6c757d; }
        #queryResult ul { list-style: none; padding-left: 0; }
        #queryResult li { margin-bottom: 5px; background-color: #f3fcf5; padding: 8px; border-left: 3px solid #28a745; }

        #stats-panel {
            margin-top: 20px;
            padding-top: 25px;
            border-top: 1px solid #eee;
        }
        #stats-panel ul {
            list-style: none;
            padding: 0;
        }
        #stats-panel li {
            margin-bottom: 10px;
            font-size: 0.95em;
        }
        .error {
            color: #dc3545;
            font-weight: bold;
            margin-top: 10px;
        }
        .message {
            color: #28a745;
            font-weight: bold;
            margin-top: 10px;
        }
        
        @media (max-width: 900px) {
            .container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <h1>GraphRL AI Agent Platform</h1>

    <div class="container">
        <div class="panel">
            <h2>Agent Management</h2>
            <div class="form-section">
                <h3>Create New Agent</h3>
                <div class="form-group">
                    <label for="agentName">Name:</label>
                    <input type="text" id="agentName" placeholder="e.g., CodeHelper">
                </div>
                <div class="form-group">
                    <label for="agentDescription">Description:</label>
                    <textarea id="agentDescription" rows="2" placeholder="e.g., An agent that helps with programming questions."></textarea>
                </div>
                <div class="form-group">
                    <label for="agentSpecialization">Specialization (keywords, comma-separated):</label>
                    <input type="text" id="agentSpecialization" placeholder="e.g., Python, JavaScript, data structures">
                </div>
                <!-- CHANGE: Replaced input with select dropdown for Ollama Model -->
                <div class="form-group">
                    <label for="agentModel">Ollama Model:</label>
                    <select id="agentModel">
                        <option value="llama2" selected>llama2</option>
                        <option value="codellama">codellama</option>
                        <option value="mistral">mistral</option>
                        <option value="neural-chat">neural-chat</option>
                        <!-- Add more models here if you pull them later -->
                    </select>
                </div>
                <!-- END CHANGE -->
                <div class="form-group">
                    <label for="systemPromptTemplate">System Prompt Template:</label>
                    <textarea id="systemPromptTemplate" rows="3" 
                        placeholder="You are {agent_name}, specialized in {specialization}. Description: {description}. Answer accurately and concisely based on your expertise. If you lack information, state it clearly."
                        title="Use {agent_name}, {specialization}, {description} as placeholders."></textarea>
                </div>
                <button class="button" onclick="createAgent()">Create Agent</button>
                <p id="createAgentMessage" class="message"></p>
            </div>

            <div class="agent-list-container">
                <h3>Existing Agents</h3>
                <div id="agentList">
                    <p>Loading agents...</p>
                </div>
            </div>
        </div>

        <div class="panel">
            <h2>Agent Interaction</h2>
            <div class="form-section">
                <h3>Query an Agent</h3>
                <div class="form-group">
                    <label for="selectAgent">Select Agent:</label>
                    <select id="selectAgent">
                        <option value="">-- Select an Agent --</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="queryText">Query:</label>
                    <textarea id="queryText" rows="4" placeholder="Enter your query here..."></textarea>
                </div>
                <button class="button" onclick="querySelectedAgent()">Send Query</button>
            </div>

            <div id="queryResult">
                <h3>Query Result</h3>
                <p>Results will appear here after you send a query.</p>
            </div>

            <div id="stats-panel">
                <h2>Platform Statistics</h2>
                <ul>
                    <li><strong>Total Agents:</strong> <span id="totalAgents">0</span></li>
                    <li><strong>Total Connections:</strong> <span id="totalConnections">0</span></li>
                    <li><strong>Total Queries:</strong> <span id="totalQueries">0</span></li>
                </ul>
                <h3>Top Collaborating Agents (Source)</h3>
                <ul id="topSourceAgents"><li>No data</li></ul>
                <h3>Top Collaborated-With Agents (Target)</h3>
                <ul id="topTargetAgents"><li>No data</li></ul>
            </div>
        </div>
    </div>

    <script>
        const API_BASE_URL = '';

        async function fetchAgents() {
            try {
                const response = await fetch(`${API_BASE_URL}/api/agents`);
                const agents = await response.json();
                const agentListDiv = document.getElementById('agentList');
                const selectAgentDropdown = document.getElementById('selectAgent');
                agentListDiv.innerHTML = '';
                selectAgentDropdown.innerHTML = '<option value="">-- Select an Agent --</option>';

                if (agents.length === 0) {
                    agentListDiv.innerHTML = '<p>No agents created yet.</p>';
                    return;
                }

                agents.forEach(agent => {
                    const card = document.createElement('div');
                    card.className = 'agent-card';
                    card.innerHTML = `
                        <h3>${agent.name}</h3>
                        <p><strong>ID:</strong> ${agent.id.substring(0, 8)}...</p>
                        <p><strong>Description:</strong> ${agent.description}</p>
                        <p><strong>Specialization:</strong> ${agent.specialization}</p>
                        <p><strong>Model:</strong> ${agent.model_config.model}</p>
                        <p><strong>Prompt:</strong> ${agent.system_prompt_template.substring(0, Math.min(agent.system_prompt_template.length, 100))}...</p>
                        <p><strong>Created:</strong> ${new Date(agent.created_at).toLocaleString()}</p>
                    `;
                    agentListDiv.appendChild(card);

                    const option = document.createElement('option');
                    option.value = agent.id;
                    option.textContent = agent.name;
                    selectAgentDropdown.appendChild(option);
                });
            } catch (error) {
                console.error('Error fetching agents:', error);
                document.getElementById('agentList').innerHTML = '<p class="error">Error loading agents. Check server console.</p>';
            }
        }

        async function createAgent() {
            const name = document.getElementById('agentName').value;
            const description = document.getElementById('agentDescription').value;
            const specialization = document.getElementById('agentSpecialization').value;
            // CHANGE: Read value from select dropdown
            const model = document.getElementById('agentModel').value;
            // END CHANGE
            const systemPromptTemplate = document.getElementById('systemPromptTemplate').value;
            const messageDiv = document.getElementById('createAgentMessage');

            messageDiv.textContent = '';
            messageDiv.className = 'message';

            if (!name || !specialization || !model) {
                messageDiv.textContent = 'Please fill in Name, Specialization, and Ollama Model.';
                messageDiv.className = 'error';
                return;
            }

            try {
                const response = await fetch(`${API_BASE_URL}/api/create_agent`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name,
                        description,
                        specialization,
                        model_config: { model },
                        system_prompt_template: systemPromptTemplate
                    })
                });
                const data = await response.json();
                if (response.ok) {
                    messageDiv.textContent = `Agent created successfully! ID: ${data.agent_id.substring(0, 8)}...`;
                    document.getElementById('agentName').value = '';
                    document.getElementById('agentDescription').value = '';
                    document.getElementById('agentSpecialization').value = '';
                    // No need to reset model input, as it's a dropdown with a default selected
                    // document.getElementById('systemPromptTemplate').value = 'You are {agent_name}...'; // Reset to default if desired
                    fetchAgents();
                    fetchStats();
                } else {
                    messageDiv.textContent = `Error: ${data.error || 'Failed to create agent.'}`;
                    messageDiv.className = 'error';
                }
            } catch (error) {
                console.error('Error creating agent:', error);
                messageDiv.textContent = 'An error occurred while creating the agent. Check server console.';
                messageDiv.className = 'error';
            }
        }

        async function querySelectedAgent() {
            const agentId = document.getElementById('selectAgent').value;
            const queryText = document.getElementById('queryText').value;
            const queryResultDiv = document.getElementById('queryResult');
            queryResultDiv.innerHTML = '<p>Sending query...</p>';
            queryResultDiv.className = 'panel';

            if (!agentId) {
                queryResultDiv.innerHTML = '<p class="error">Please select an agent.</p>';
                return;
            }
            if (!queryText) {
                queryResultDiv.innerHTML = '<p class="error">Please enter a query.</p>';
                return;
            }

            try {
                const response = await fetch(`${API_BASE_URL}/api/agent/${agentId}/query`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: queryText })
                });
                const data = await response.json();
                if (response.ok) {
                    let resultHtml = `<h3>Query Result from ${document.getElementById('selectAgent').options[document.getElementById('selectAgent').selectedIndex].text}</h3>`;
                    resultHtml += `<p><strong>Query:</strong> ${data.query}</p>`;
                    resultHtml += `<p><strong>Response:</strong> ${data.response}</p>`;
                    resultHtml += `<p><strong>Confidence:</strong> <span style="font-weight: bold; color: ${data.confidence < 0.5 ? '#dc3545' : '#28a745'};">${data.confidence.toFixed(2)}</span></p>`;
                    resultHtml += `<p><strong>Collaboration Used:</strong> <span style="font-weight: bold; color: ${data.collaboration_used ? '#28a745' : '#6c757d'};">${data.collaboration_used ? 'Yes' : 'No'}</span></p>`;

                    if (data.collaboration_used && data.collaborative_responses && data.collaborative_responses.length > 0) {
                        resultHtml += `<h4>Collaborating Agents:</h4><ul>`;
                        data.collaborative_responses.forEach(collab => {
                            resultHtml += `<li><strong>${collab.agent_name}</strong> (Similarity: ${collab.similarity.toFixed(2)}): ${collab.response.substring(0, Math.min(collab.response.length, 200))}...</li>`;
                        });
                        resultHtml += `</ul>`;
                    }
                    queryResultDiv.innerHTML = resultHtml;
                    fetchStats();
                } else {
                    queryResultDiv.innerHTML = `<p class="error">Error: ${data.error || 'Failed to query agent.'}</p>`;
                }
            } catch (error) {
                console.error('Error querying agent:', error);
                queryResultDiv.innerHTML = '<p class="error">An error occurred while querying the agent. Check server console and ensure Ollama is running and has the selected model pulled.</p>';
            }
        }

        async function fetchStats() {
            try {
                const response = await fetch(`${API_BASE_URL}/api/stats`);
                const stats = await response.json();

                document.getElementById('totalAgents').textContent = stats.total_agents;
                document.getElementById('totalConnections').textContent = stats.total_connections;
                document.getElementById('totalQueries').textContent = stats.total_queries;

                const topSourceAgentsList = document.getElementById('topSourceAgents');
                topSourceAgentsList.innerHTML = '';
                if (stats.top_source_agents.length > 0) {
                    stats.top_source_agents.forEach(item => {
                        topSourceAgentsList.innerHTML += `<li>${item.agent_name}: ${item.count}</li>`;
                    });
                } else {
                    topSourceAgentsList.innerHTML = '<li>No data</li>';
                }

                const topTargetAgentsList = document.getElementById('topTargetAgents');
                topTargetAgentsList.innerHTML = ''; // Fixed typo here (was topTargetAgents without List)
                if (stats.top_target_agents.length > 0) {
                    stats.top_target_agents.forEach(item => {
                        topTargetAgentsList.innerHTML += `<li>${item.agent_name}: ${item.count}</li>`;
                    });
                } else {
                    topTargetAgentsList.innerHTML = '<li>No data</li>';
                }

            } catch (error) {
                console.error('Error fetching stats:', error);
            }
        }

        document.addEventListener('DOMContentLoaded', () => {
            fetchAgents();
            fetchStats();
            setInterval(fetchStats, 5001);
        });
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    print("\n--- Performing initial Ollama server check ---")
    if not _check_ollama_status_internal(OLLAMA_HOST, OLLAMA_TIMEOUT):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARNING: Ollama server is not reachable at {OLLAMA_HOST}. Some functionalities might be impaired.")
        print("Please ensure `ollama serve` is running in your terminal and the selected models are pulled (`ollama pull model_name`).")
    print("----------------------------------------------\n")

    try:
        if not os.path.exists('index.html'):
            with open("index.html", "w") as f:
                f.write(html_template)
            print("Generated index.html successfully.")
    except IOError as e:
        print(f"Error writing index.html: {e}")
        print("Please ensure you have write permissions in the current directory.")

    print("Starting Flask app...")
    app.run(debug=True, port=5001, use_reloader=False)
