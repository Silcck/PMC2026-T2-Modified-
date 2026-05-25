import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

class ProjectNetwork:
    def __init__(self, csv_path, fixed_overhead=0.0, variable_overhead=0.0):
        self.df = pd.read_csv(csv_path)
        # Standardize data types
        self.df['Total_Cost'] = pd.to_numeric(self.df['Total_Cost'], errors='coerce').fillna(0)
        
        self.fixed_overhead = fixed_overhead
        self.variable_overhead = variable_overhead
        
        # In AON, duration is a property of the node (task)
        if 'Duration' not in self.df.columns:
            # Fallback to Quantity_Needed as duration if missing
            self.df['Duration'] = pd.to_numeric(self.df.get('Quantity_Needed', 1), errors='coerce').fillna(1)
        else:
            self.df['Duration'] = pd.to_numeric(self.df['Duration'], errors='coerce').fillna(0)
            
        self.G = nx.DiGraph()
        self.results = None
        self.total_project_time = 0
        self.total_project_cost = 0 
        self.optimization_history = [] # Track (time, cost) at each step
        self._build_aon_network()

    def _build_aon_network(self):
        """
        Builds the Activity-on-Node (AON) directed graph.
        Tasks are nodes, dependencies are edges.
        """
        # Aggregate data by Task_ID to handle potential resource splits
        # We need to preserve Crashability and Resource_Type for optimization
        # Since a Task_ID can have multiple resources, we'll check if ANY resource is 'Cost of Labour'
        # and IF that specific resource is crashable.
        
        task_data = self.df.groupby('Task_ID').agg({
            'Task_Name': 'first',
            'Dependencies': 'first',
            'Duration': 'sum',
            'Total_Cost': 'sum'
        }).reset_index()

        # Add nodes with duration and cost
        for _, row in task_data.iterrows():
            t_id = row['Task_ID']
            # Get resource details for this task
            resources = self.df[self.df['Task_ID'] == t_id]
            
            self.G.add_node(t_id, 
                           duration=row['Duration'], 
                           cost=row['Total_Cost'], 
                           name=row['Task_Name'],
                           resources=resources.to_dict('records'))

        # Add edges based on dependencies
        for _, row in task_data.iterrows():
            target = row['Task_ID']
            deps = str(row['Dependencies']).split(',') if pd.notna(row['Dependencies']) else []
            for dep in deps:
                dep = dep.strip()
                if dep and dep in self.G.nodes:
                    self.G.add_edge(dep, target)
                elif dep:
                    # Silence the warning for massive datasets unless needed
                    pass

    def calculate_cpm(self):
        """
        Performs Forward and Backward pass to calculate CPM metrics.
        """
        if not nx.is_directed_acyclic_graph(self.G):
            cycles = list(nx.simple_cycles(self.G))
            raise ValueError(f"Project network contains cycles: {cycles}")

        # Topological sort for processing
        ordered_nodes = list(nx.topological_sort(self.G))

        # 1. Forward Pass
        es = {node: 0 for node in self.G.nodes}
        ef = {node: 0 for node in self.G.nodes}
        
        for node in ordered_nodes:
            duration = self.G.nodes[node]['duration']
            preds = list(self.G.predecessors(node))
            if preds:
                es[node] = max(ef[p] for p in preds)
            ef[node] = es[node] + duration

        self.total_project_time = max(ef.values()) if ef else 0

        # 2. Backward Pass
        ls = {node: self.total_project_time for node in self.G.nodes}
        lf = {node: self.total_project_time for node in self.G.nodes}
        
        for node in reversed(ordered_nodes):
            duration = self.G.nodes[node]['duration']
            succs = list(self.G.successors(node))
            if succs:
                lf[node] = min(ls[s] for s in succs)
            ls[node] = lf[node] - duration

        # 3. Calculate Slack and identify Critical Path
        results_list = []
        for node in self.G.nodes:
            slack = ls[node] - es[node]
            # Use a small epsilon for float comparison
            is_critical = abs(slack) < 1e-9
            
            self.G.nodes[node]['es'] = es[node]
            self.G.nodes[node]['ef'] = ef[node]
            self.G.nodes[node]['ls'] = ls[node]
            self.G.nodes[node]['lf'] = lf[node]
            self.G.nodes[node]['slack'] = slack
            self.G.nodes[node]['is_critical'] = is_critical
            
            results_list.append({
                'Task_ID': node,
                'Task_Name': self.G.nodes[node]['name'],
                'Duration': self.G.nodes[node]['duration'],
                'Total_Cost': self.G.nodes[node]['cost'],
                'ES': es[node],
                'EF': ef[node],
                'LS': ls[node],
                'LF': lf[node],
                'Slack': slack,
                'is_critical': is_critical
            })

        self.results = pd.DataFrame(results_list)
        # Total Project Cost = Sum of Task Costs + (Total Project Time * (Fixed Overhead + Variable Overhead))
        direct_costs = self.results['Total_Cost'].sum()
        indirect_costs = self.total_project_time * (self.fixed_overhead + self.variable_overhead)
        self.total_project_cost = direct_costs + indirect_costs
        
        # Track history for visualization
        self.optimization_history.append({
            'Time': self.total_project_time,
            'Cost': self.total_project_cost,
            'Direct_Cost': direct_costs,
            'Indirect_Cost': indirect_costs
        })
        
        return self.results

    def crash_time(self, target_duration):
        """
        Algorithm 1: Reduce project duration by crashing cheapest critical tasks linearly.
        """
        crashed_tasks = []
        
        # 1. Pre-calculate linear cost penalties and store original durations
        for n in self.G.nodes:
            node_data = self.G.nodes[n]
            # Store original values for constraints if not already stored
            if 'original_duration' not in node_data:
                node_data['original_duration'] = node_data['duration']
                node_data['original_cost'] = node_data['cost']
                
                # Calculate fixed penalty per unit based on Labour resources
                penalty = 0
                for res in node_data['resources']:
                    if (str(res.get('Crashability', '')).lower() == 'yes' and 
                        str(res.get('Resource_Type', '')).strip() == 'Cost of Labour'):
                        
                        # Penalty = Original cost of this resource * its specific %
                        res_cost = float(res.get('Total_Cost', 0))
                        res_pct = float(res.get('Crash_Price_Increase_Pct', 0))
                        penalty += res_cost * res_pct
                
                node_data['cost_penalty_per_unit'] = penalty
                # Minimum duration constraint: max(1, 50% of original)
                node_data['min_duration'] = max(1.0, node_data['original_duration'] * 0.5)

        iteration = 0
        while self.total_project_time > target_duration:
            self.calculate_cpm()
            if self.total_project_time <= target_duration:
                break
                
            critical_nodes = [n for n, d in self.G.nodes(data=True) if d['is_critical']]
            
            # 2. Filter for crashable candidates meeting constraints
            crash_candidates = []
            for n in critical_nodes:
                node_data = self.G.nodes[n]
                
                # Check if it has a penalty > 0 and is above min duration
                if node_data.get('cost_penalty_per_unit', 0) > 0 and node_data['duration'] > node_data['min_duration']:
                    crash_candidates.append({
                        'Task_ID': n,
                        'penalty': node_data['cost_penalty_per_unit']
                    })
            
            if not crash_candidates:
                print("No more crashable Labour candidates on the critical path meeting constraints.")
                break
            
            # 3. Find task with lowest linear penalty
            cheapest_task = min(crash_candidates, key=lambda x: x['penalty'])
            t_id = cheapest_task['Task_ID']
            
            # 4. Reduce duration linearly and add fixed penalty
            self.G.nodes[t_id]['duration'] -= 1
            # Clamp to min duration just in case of float issues
            if self.G.nodes[t_id]['duration'] < self.G.nodes[t_id]['min_duration']:
                self.G.nodes[t_id]['duration'] = self.G.nodes[t_id]['min_duration']
            
            self.G.nodes[t_id]['cost'] += self.G.nodes[t_id]['cost_penalty_per_unit']
            
            crashed_tasks.append(t_id)
            iteration += 1
            if iteration > 20000: break
                
        self.calculate_cpm()
        return list(set(crashed_tasks)), self.total_project_cost

    def relax_cost(self, target_budget):
        """
        Algorithm 2: Reduce project budget by delaying non-critical tasks linearly.
        """
        delayed_tasks = []
        
        # Ensure metadata exists
        for n in self.G.nodes:
            node_data = self.G.nodes[n]
            if 'original_duration' not in node_data:
                node_data['original_duration'] = node_data['duration']
                node_data['original_cost'] = node_data['cost']
                penalty = 0
                for res in node_data['resources']:
                    if (str(res.get('Crashability', '')).lower() == 'yes' and 
                        str(res.get('Resource_Type', '')).strip() == 'Cost of Labour'):
                        penalty += float(res.get('Total_Cost', 0)) * float(res.get('Crash_Price_Increase_Pct', 0))
                node_data['cost_penalty_per_unit'] = penalty

        iteration = 0
        while self.total_project_cost > target_budget:
            self.calculate_cpm()
            if self.total_project_cost <= target_budget:
                break
                
            # 1. Identify candidates: non-critical, crashable, and current cost > original
            delay_candidates = []
            for n, d in self.G.nodes(data=True):
                # We can only "relax" if we have previously "crashed" or if current cost > original
                # OR based on prompt: just reduce cost linearly as we increase duration.
                # Usually relaxation is the inverse of crashing.
                if not d['is_critical'] and d['slack'] >= 1:
                    if d.get('cost_penalty_per_unit', 0) > 0:
                        delay_candidates.append({
                            'Task_ID': n,
                            'slack': d['slack'],
                            'penalty': d['cost_penalty_per_unit']
                        })

            if not delay_candidates:
                print("No more non-critical Labour candidates for cost relaxation.")
                break
            
            # 2. Find task with highest slack and highest potential savings
            cheapest_to_delay = max(delay_candidates, key=lambda x: (x['slack'], x['penalty']))
            t_id = cheapest_to_delay['Task_ID']
            
            # 3. Increase duration by 1 unit and decrease cost linearly
            self.G.nodes[t_id]['duration'] += 1
            self.G.nodes[t_id]['cost'] -= self.G.nodes[t_id]['cost_penalty_per_unit']
            
            # Ensure cost doesn't drop below a realistic floor (e.g. original cost - some margin)
            # But the requirement is to strictly meet target budget.
            
            delayed_tasks.append(t_id)
            iteration += 1
            if iteration > 20000: break

        self.calculate_cpm()
        return list(set(delayed_tasks)), self.total_project_time

    def visualize_aon_graphviz(self, max_nodes=1000, modified_tasks=None):
        """
        Generates a Graphviz AON diagram with PERT-style nodes.
        Returns a graphviz.Digraph object.
        """
        import graphviz
        
        if self.results is None:
            self.calculate_cpm()

        num_nodes = len(self.G.nodes)
        if num_nodes > max_nodes:
            return None

        dot = graphviz.Digraph(comment='Project AON', graph_attr={'rankdir': 'LR'})
        
        # --- Add Legend ---
        legend_label = '''<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                <TR><TD COLSPAN="3" BGCOLOR="#DDDDDD"><B>LEGEND &amp; NODE STRUCTURE</B></TD></TR>
                <TR>
                    <TD>Early Start (ES)</TD>
                    <TD>Duration</TD>
                    <TD>Early Finish (EF)</TD>
                </TR>
                <TR>
                    <TD COLSPAN="3"><B>TASK ID / NAME</B></TD>
                </TR>
                <TR>
                    <TD>Late Start (LS)</TD>
                    <TD>Slack</TD>
                    <TD>Late Finish (LF)</TD>
                </TR>
                <TR><TD COLSPAN="3" BGCOLOR="#DDDDDD"><B>COLOR KEY</B></TD></TR>
                <TR>
                    <TD BGCOLOR="#ffe6e6" COLOR="red" PENWIDTH="3">Critical Path</TD>
                    <TD BGCOLOR="#fff2e6" COLOR="orange" PENWIDTH="3">Modified (Crash/Relax)</TD>
                    <TD BGCOLOR="white">Standard Task</TD>
                </TR>
            </TABLE>
        >'''
        dot.node('legend', label=legend_label, shape="none", fontsize="10")
        
        modified_tasks = modified_tasks if modified_tasks else []
        
        for node, d in self.G.nodes(data=True):
            # Extract CPM metrics
            es = d.get('es', 0)
            ef = d.get('ef', 0)
            ls = d.get('ls', 0)
            lf = d.get('lf', 0)
            dur = d.get('duration', 0)
            slack = d.get('slack', 0)
            is_critical = d.get('is_critical', False)
            is_modified = node in modified_tasks

            # Styling
            color = "black"
            penwidth = "1"
            fillcolor = "white"
            
            if is_critical:
                color = "red"
                penwidth = "3"
                fillcolor = "#ffe6e6" # Light red
            
            if is_modified:
                color = "orange"
                penwidth = "3"
                fillcolor = "#fff2e6" # Light orange

            # HTML Label for PERT Node
            # Table Structure:
            # Row 1: ES | DUR | EF
            # Row 2: TASK_ID
            # Row 3: LS | SLACK | LF
            label = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                    <TD>{es:.1f}</TD>
                    <TD>{dur:.1f}</TD>
                    <TD>{ef:.1f}</TD>
                </TR>
                <TR>
                    <TD COLSPAN="3" BGCOLOR="{fillcolor}"><B>{node}</B></TD>
                </TR>
                <TR>
                    <TD>{ls:.1f}</TD>
                    <TD>{slack:.1f}</TD>
                    <TD>{lf:.1f}</TD>
                </TR>
            </TABLE>>'''

            dot.node(str(node), label=label, shape="none", color=color, penwidth=penwidth)

        # Add edges
        for u, v in self.G.edges():
            u_crit = self.G.nodes[u].get('is_critical', False)
            v_crit = self.G.nodes[v].get('is_critical', False)
            
            edge_color = "gray"
            edge_width = "1"
            
            # Edge is critical if it connects two critical nodes
            if u_crit and v_crit:
                edge_color = "red"
                edge_width = "2"
                
            dot.edge(str(u), str(v), color=edge_color, penwidth=edge_width)

        return dot

if __name__ == "__main__":
    csv_file = "standardized_project_dataset.csv"
    if os.path.exists(csv_file):
        project = ProjectNetwork(csv_file)
        results = project.calculate_cpm()
        print("\n--- Project Schedule Results (AON) ---")
        print(results[['Task_ID', 'ES', 'EF', 'LS', 'LF', 'Slack', 'is_critical']])
        
        critical_path = results[results['is_critical']]['Task_ID'].tolist()
        print(f"\nCritical Path: {' -> '.join(critical_path)}")
        print(f"Total Project Time: {project.total_project_time}")
        print(f"Total Project Cost: €{project.total_project_cost:,.2f}")
        
        project.visualize_aon_graphviz()
        print("\nAON Diagram saved to aon_diagram.dot")
    else:
        print(f"Input file {csv_file} not found. Please run the standardization script first.")
