import hashlib
import json
import time
from typing import List, Dict, Any
from flask import Flask, jsonify, request
from uuid import uuid4
from urllib.parse import urlparse

class Transaction:
    def __init__(self, sender: str, recipient: str, amount: float):
        self.sender = sender
        self.recipient = recipient
        self.amount = amount

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sender': self.sender,
            'recipient': self.recipient,
            'amount': self.amount
        }


class Block:
    def __init__(self, index: int, previous_hash: str, timestamp: float, transactions: List[Transaction]):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.transactions = transactions
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        block_string = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'hash': self.hash
        }


class Blockchain:
    def __init__(self):
        self.chain: List[Block] = []
        self.current_transactions: List[Transaction] = []
        self.create_block(previous_hash='1')  # Create genesis block

    def create_block(self, previous_hash: str) -> Block:
        block = Block(index=len(self.chain) + 1,
                      previous_hash=previous_hash,
                      timestamp=time.time(),
                      transactions=self.current_transactions)
        self.current_transactions = []
        self.chain.append(block)
        return block

    def add_transaction(self, sender: str, recipient: str, amount: float) -> int:
        transaction = Transaction(sender, recipient, amount)
        self.current_transactions.append(transaction)
        return self.last_block.index + 1

    @property
    def last_block(self) -> Block:
        return self.chain[-1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'chain': [block.to_dict() for block in self.chain],
            'length': len(self.chain)
        }


class Node:
    def __init__(self):
        self.blockchain = Blockchain()
        self.nodes: set = set()

    def register_node(self, address: str):
        parsed_uri = urlparse(address)
        self.nodes.add(parsed_uri.netloc)

    def resolve_conflicts(self) -> bool:
        # Placeholder for conflict resolution logic (like Proof of Work)
        return False


app = Flask(__name__)
node_identifier = str(uuid4()).replace('-', '')
node = Node()


@app.route('/transactions/new', methods=['POST'])
def new_transaction():
    values = request.get_json()
    required = ['sender', 'recipient', 'amount']
    if not all(k in values for k in required):
        return 'Missing values', 400

    index = node.blockchain.add_transaction(values['sender'], values['recipient'], values['amount'])
    response = {'message': f'Transaction will be added to Block {index}'}
    return jsonify(response), 201


@app.route('/mine', methods=['GET'])
def mine():
    last_block = node.blockchain.last_block
    node.blockchain.create_block(previous_hash=last_block.hash)

    response = {
        'message': 'New Block Forged',
        'index': last_block.index + 1,
        'transactions': [tx.to_dict() for tx in last_block.transactions]
    }
    return jsonify(response), 201


@app.route('/chain', methods=['GET'])
def full_chain():
    response = node.blockchain.to_dict()
    return jsonify(response), 200


@app.route('/nodes/register', methods=['POST'])
def register_nodes():
    values = request.get_json()
    nodes = values.get('nodes')
    if nodes is None:
        return 'Error: Please supply a valid list of nodes', 400

    for node in nodes:
        node.register_node(node)

    response = {
        'message': 'New nodes have been added',
        'total_nodes': list(node.nodes),
    }
    return jsonify(response), 201


@app.route('/nodes/resolve', methods=['GET'])
def consensus():
    replaced = node.resolve_conflicts()
    if replaced:
        response = {
            'message': 'Our chain was replaced',
            'new_chain': node.blockchain.to_dict()
        }
    else:
        response = {
            'message': 'Our chain is authoritative',
            'chain': node.blockchain.to_dict()
        }
    return jsonify(response), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)