import json
import os
from datetime import datetime
import time # For timestamp in reflection records

class EnhancedAIMDictionary:
    def __init__(self, file_path="enhanced_aim_dictionary.json"):
        self.file_path = file_path
        self.aim_entries = {}  # {aim_id: {aim_sequence, human_label, first_seen_round, contexts, evolution_trace}}
        self.reflection_records = {}  # {aim_id: [reflection_data_list]}
        self.unified_records_list = [] # Store all unified records for analysis later

        self._load_from_file() # Load existing data

    def _load_from_file(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    data = json.load(f)
                    self.aim_entries = data.get('aim_entries', {})
                    self.reflection_records = data.get('reflection_records', {})
                    self.unified_records_list = data.get('unified_records_list', [])
                    print(f"[EnhancedAIMDictionary] Loaded {len(self.aim_entries)} AIM entries and {len(self.unified_records_list)} unified records from {self.file_path}")
            except json.JSONDecodeError:
                print(f"Warning: {self.file_path} is corrupted or empty. Starting with empty dictionary.")

    def _generate_aim_id(self, aim_sequence, context, round_num, agent_id):
        # Create a more descriptive AIM_ID, combining sequence, round, agent ID and partial context
        # Since aim_sequence is a list, use str(aim_sequence) directly
        # context can be a dict, here simplified to a string format
        context_str = str(context.get('context', '')) # Ensure context is a string, preventing complex objects from being part of ID
        return f"{agent_id}_R{round_num}_{str(aim_sequence)}"

    def add_entry_with_reflection(self, aim_sequence, human_interpretation, context, 
                                 agent_reflection_data, round_num, agent_id):
        
        aim_id = self._generate_aim_id(aim_sequence, context, round_num, agent_id)
        
        if aim_id not in self.aim_entries:
            self.aim_entries[aim_id] = {
                'aim_sequence': aim_sequence,
                'human_label': human_interpretation,
                'first_seen_round': round_num,
                'contexts': [], # List of dicts: {'context_detail': ..., 'round': ..., 'success_rate': ..., 'semantic_stability': ...}
                'evolution_trace': [] # Can be used to track the evolution of AIM semantics
            }
        
        # Add current context details (simplify success_rate and semantic_stability as placeholders)
        current_context_detail = {
            'context_detail': context.get('context_detail', context), # Try to get detailed context, otherwise use raw context
            'round': round_num,
            'success_rate': self._compute_success_rate(aim_id, round_num), # Simplified calculation
            'semantic_stability': self._compute_semantic_stability(aim_id, round_num) # Simplified calculation
        }
        self.aim_entries[aim_id]['contexts'].append(current_context_detail)
        
        # Link reflection records
        if aim_id not in self.reflection_records:
            self.reflection_records[aim_id] = []
            
        self.reflection_records[aim_id].append({
            'round': round_num,
            'agent_id': agent_id,
            'reflection_data': agent_reflection_data,
            'human_label': human_interpretation,
            'timestamp': datetime.now().isoformat() # Use ISO format string
        })
        
        return aim_id

    def add_unified_record(self, record):
        """Add a complete unified record to facilitate later analysis"""
        self.unified_records_list.append(record)

    def _compute_success_rate(self, aim_id, current_round):
        # This is a simplified calculation, actually need to track the behavioral outcome (reward) of each AIM
        # Suppose it is successful if interpreted as C and reward is high, or D and reward is high
        # For DEMO purposes, return a simulated value
        return round(0.5 + 0.5 * (current_round % 100) / 100, 2) # Simulate change over time

    def _compute_semantic_stability(self, aim_id, current_round):
        # Simplified calculation: based on the consistency of the historical human_label of a specific AIM_ID
        # Actually need to iterate through self.reflection_records[aim_id] to compute the distribution of human_label
        # For DEMO purposes, return a simulated value
        return round(0.7 + 0.3 * (current_round % 50) / 50, 2) # Simulate change over time

    def save(self):
        data_to_save = {
            'aim_entries': self.aim_entries,
            'reflection_records': self.reflection_records,
            'unified_records_list': self.unified_records_list
        }
        with open(self.file_path, "w") as f:
            json.dump(data_to_save, f, indent=2)
        print(f"[EnhancedAIMDictionary] Saved {len(self.aim_entries)} AIM entries and {len(self.unified_records_list)} unified records to {self.file_path}")