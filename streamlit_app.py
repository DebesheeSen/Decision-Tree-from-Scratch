# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch
from collections import Counter
import io
import sys
from decision_tree import DecisionTree
from graphviz import Digraph

def calculate_tree_width(node):
    """Calculate the width needed for each subtree"""
    if node.is_leaf_node():
        return 1
    
    if not hasattr(node, 'branches') or not node.branches:
        return 1
    
    total_width = sum(calculate_tree_width(branch) for branch in node.branches.values())
    return max(total_width, len(node.branches))

def visualize_tree_improved(node, feature_names, graph=None, parent=None, edge_label=None, 
                          pos_dict=None, level=0, position=0, spacing=2.0):
    """
    Improved tree visualization with better spacing to prevent overlapping
    """
    if graph is None:
        graph = nx.DiGraph()
        pos_dict = {}
    
    # Create node label
    if node.is_leaf_node():
        # Decode the target value if target_encoder exists
        if hasattr(st.session_state, 'target_encoder'):
            class_value = [k for k, v in st.session_state.target_encoder.items() if v == node.value][0]
        else:
            class_value = node.value
        node_label = f"Class:\n{class_value}"
        node_color = "#90EE90" if node.value == 1 else "#FFB6C1"
        node_shape = "circle"
    else:
        node_label = f"{feature_names[node.feature_index]}"
        node_color = "#87CEEB"
        node_shape = "box"
    
    # Add node to graph
    node_id = id(node)
    graph.add_node(node_id, 
                  label=node_label, 
                  color=node_color, 
                  shape=node_shape,
                  level=level)
    
    # Calculate position with proper spacing
    pos_dict[node_id] = (position, -level * spacing)
    
    # Add edge from parent
    if parent is not None:
        graph.add_edge(parent, node_id, label=str(edge_label))
    
    # Recursively add children with better spacing
    if not node.is_leaf_node():
        branches = list(node.branches.items())
        num_branches = len(branches)
        
        if num_branches == 1:
            # Single child: place directly below
            branch_positions = [position]
        else:
            # Multiple children: calculate proper spacing based on subtree widths
            widths = [calculate_tree_width(branch) for _, branch in branches]
            total_width = sum(widths)
            
            # Calculate spacing factor based on tree depth and width
            base_spacing = max(1.5, total_width * 0.8)
            
            # Create positions with proportional spacing
            positions = []
            current_pos = position - (total_width - 1) * base_spacing / 2
            
            for i, width in enumerate(widths):
                if i == 0:
                    pos = current_pos
                else:
                    pos = current_pos + (widths[i-1] + width) * base_spacing / 2
                positions.append(pos)
                current_pos = pos
            
            branch_positions = positions
        
        for i, (val, branch) in enumerate(branches):
            feature_name = feature_names[node.feature_index]
            if feature_name in st.session_state.label_encoders:
                # Handle dictionary-style encoding
                val_label = [k for k, v in st.session_state.label_encoders[feature_name].items() if v == val][0]
            else:
                val_label = str(val)
                
            visualize_tree_improved(branch, feature_names, graph, node_id, val_label, 
                                  pos_dict, level + 1, branch_positions[i], spacing)
    
    return graph, pos_dict

def plot_tree_beautiful(node, feature_names, title="Decision Tree", figsize=(18, 12)):
    """
    Create a beautiful tree plot with custom styling and no overlapping
    """
    # Build the graph with better spacing
    G, pos = visualize_tree_improved(node, feature_names)
    
    # Create figure with larger size
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Calculate dynamic margins based on tree spread
    x_coords = [x for x, y in pos.values()]
    y_coords = [y for x, y in pos.values()]
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    
    # Get node and edge attributes
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]
    node_labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'label')
    
    # Draw edges first (so they appear behind nodes)
    nx.draw_networkx_edges(G, pos, edge_color='#666666', 
                          arrows=True, arrowsize=25, arrowstyle='->', 
                          width=2.5, alpha=0.7, ax=ax)
    
    # Draw nodes with adaptive sizing
    node_size = max(0.25, min(0.4, 8.0 / (len(pos) ** 0.5)))  # Adaptive node size
    
    for node, (x, y) in pos.items():
        color = G.nodes[node]['color']
        label = G.nodes[node]['label']
        shape = G.nodes[node]['shape']
        
        if shape == 'circle':  # Leaf nodes
            circle = plt.Circle((x, y), node_size, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, label, ha='center', va='center', 
                   fontsize=max(8, int(12 - len(pos)/10)), fontweight='bold',
                   wrap=True)
        else:  # Internal nodes
            box_width = node_size * 2.2
            box_height = node_size * 1.4
            bbox = FancyBboxPatch((x-box_width/2, y-box_height/2), box_width, box_height, 
                                boxstyle="round,pad=0.05", 
                                facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(bbox)
            ax.text(x, y, label, ha='center', va='center', 
                   fontsize=max(9, int(13 - len(pos)/8)), fontweight='bold')
    
    # Draw edge labels with better positioning
    for edge, label in edge_labels.items():
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]
        
        # Position label closer to parent node
        t = 0.3  # Parameter to control label position along edge
        x_mid = x1 + t * (x2 - x1)
        y_mid = y1 + t * (y2 - y1)
        
        # Offset label slightly to avoid edge overlap
        offset_x = 0.1 if x2 > x1 else -0.1
        
        ax.text(x_mid + offset_x, y_mid + 0.15, label, ha='center', va='center', 
               fontsize=max(8, int(10 - len(pos)/15)), 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', 
                        alpha=0.9, edgecolor='orange', linewidth=1))
    
    # Set title and clean up plot with dynamic margins
    ax.set_title(title, fontsize=18, fontweight='bold', pad=30)
    
    # Dynamic margins based on tree size
    margin_x = max(1.5, x_range * 0.15)
    margin_y = max(0.8, y_range * 0.1)
    
    ax.set_xlim(min(x_coords) - margin_x, max(x_coords) + margin_x)
    ax.set_ylim(min(y_coords) - margin_y, max(y_coords) + margin_y)
    ax.axis('off')
    
    # Remove aspect ratio constraint to allow flexible sizing
    plt.tight_layout()
    return fig, ax


def plot_tree_graphviz(node, feature_names, title="Decision Tree"):
    """Create a tree visualization using Graphviz"""
    dot = Digraph(comment=title)
    dot.attr('node', shape='box', style='filled', color='lightgrey')
    dot.attr('edge', color='black')
    
    def add_nodes_edges(node, parent_id=None, edge_label=None):
        # Create unique node ID
        node_id = str(id(node))
        
        # Set node label and style
        if node.is_leaf_node():
            if hasattr(st.session_state, 'target_encoder'):
                class_name = [k for k, v in st.session_state.target_encoder.items() if v == node.value][0]
            else:
                class_name = str(node.value)
            node_label = f"Predict: {class_name}"
            dot.node(node_id, label=node_label, shape='ellipse', color='lightblue2')
        else:
            feature_name = feature_names[node.feature_index]
            dot.node(node_id, label=feature_name)
        
        # Add edge from parent if exists
        if parent_id is not None:
            dot.edge(parent_id, node_id, label=str(edge_label))
        
        # Add children if not leaf
        if not node.is_leaf_node():
            for val, branch in node.branches.items():
                feature_name = feature_names[node.feature_index]
                if feature_name in st.session_state.label_encoders:
                    val_label = [k for k, v in st.session_state.label_encoders[feature_name].items() if v == val][0]
                else:
                    val_label = str(val)
                add_nodes_edges(branch, node_id, val_label)
    
    # Build the graph
    add_nodes_edges(node)
    
    # Return the graph object
    return dot


def calculate_feature_importance(tree, feature_names, X, y):
    """Calculate simple feature importance based on tree structure"""
    importance = {name: 0 for name in feature_names}
    
    def traverse_importance(node, depth_weight=1.0):
        if not node.is_leaf_node():
            feature_name = feature_names[node.feature_index]
            importance[feature_name] += depth_weight
            
            for branch in node.branches.values():
                traverse_importance(branch, depth_weight * 0.8)
    
    traverse_importance(tree.root)
    
    # Normalize
    total = sum(importance.values())
    if total > 0:
        importance = {k: v/total for k, v in importance.items()}
    
    return importance

# Streamlit App
def main():
    st.set_page_config(page_title="Decision Tree Builder", layout="wide")
    
    st.title("🌳 Interactive Decision Tree Builder")
    st.markdown("Build and visualize decision trees with your data!")
    
    # Sidebar for parameters
    st.sidebar.header("Tree Parameters")
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 5)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
    criterion = st.sidebar.selectbox("Criterion", ["entropy", "gini"])
    
    # Data upload section
    st.header("📊 Data Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Feature and target selection
        st.subheader("🎯 Feature Selection")
        columns = df.columns.tolist()
        
        target_column = st.selectbox("Select Target Column", columns)
        feature_columns = st.multiselect("Select Feature Columns", 
                                       [col for col in columns if col != target_column],
                                       default=[col for col in columns if col != target_column])
        
        if len(feature_columns) > 0:
            # Prepare data
            X = df[feature_columns]
            y = df[target_column]
            
            # Handle categorical variables (simple label encoding)
            X_encoded = X.copy()
            y_encoded = y.copy()
            
            # Encode features
            label_encoders = {}
            for col in feature_columns:
                if X[col].dtype == 'object':
                    unique_vals = X[col].unique()
                    label_encoders[col] = {val: i for i, val in enumerate(unique_vals)}
                    X_encoded[col] = X[col].map(label_encoders[col])
            
            # Encode target
            if y.dtype == 'object':
                unique_targets = y.unique()
                target_encoder = {val: i for i, val in enumerate(unique_targets)}
                y_encoded = y.map(target_encoder)
            
            # Store encoders in session state
            st.session_state.label_encoders = label_encoders
            if y.dtype == 'object':
                st.session_state.target_encoder = target_encoder
            
            # Train model
            if st.button("🚀 Build Decision Tree"):
                with st.spinner("Building decision tree..."):
                    # Create and train model
                    model = DecisionTree(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        criterion=criterion
                    )
                    
                    model.fit(X_encoded.values, y_encoded.values)
                    
                    # Store in session state
                    st.session_state.model = model
                    st.session_state.feature_columns = feature_columns
                    st.session_state.X_encoded = X_encoded
                    st.session_state.y_encoded = y_encoded
                    
                    st.success("✅ Decision tree built successfully!")
            
            # Display results if model exists
            if 'model' in st.session_state:
                model = st.session_state.model
                feature_columns = st.session_state.feature_columns
                
                # Tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["🌳 Tree Visualization", "📈 Model Performance", "🔍 Feature Importance", "🎯 Make Predictions"])
                
                with tab1:
                    st.subheader("Decision Tree Visualization")
                    
                    viz_type = st.radio("Choose visualization type:", 
                                       ["Text-based", "Matplotlib (Improved)"])
                    
                    if viz_type == "Text-based":
                        try:
                            # Try using Graphviz first
                            from graphviz import Digraph
                            dot = plot_tree_graphviz(model.root, feature_columns)
                            st.graphviz_chart(dot)
                        except ImportError:
                            # Fall back to matplotlib if Graphviz not available
                            st.warning("Graphviz not available, using matplotlib visualization")
                    else:
                        fig, ax = plot_tree_beautiful(model.root, feature_columns)
                        st.pyplot(fig)
                
                with tab2:
                    st.subheader("Model Performance")
                    
                    # Training accuracy
                    y_pred = model.predict(st.session_state.X_encoded.values)
                    accuracy = np.mean(y_pred == st.session_state.y_encoded.values)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Training Accuracy", f"{accuracy:.2%}")
                    
                    # Confusion matrix
                    if len(np.unique(st.session_state.y_encoded)) == 2:  # Binary classification
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(st.session_state.y_encoded, y_pred)
                        
                        fig, ax = plt.subplots(figsize=(6, 4))
                        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                        ax.figure.colorbar(im, ax=ax)
                        
                        classes = np.unique(st.session_state.y_encoded)
                        ax.set(xticks=np.arange(cm.shape[1]),
                              yticks=np.arange(cm.shape[0]),
                              xticklabels=classes, yticklabels=classes,
                              title='Confusion Matrix',
                              ylabel='True label',
                              xlabel='Predicted label')
                        
                        # Add text annotations
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                ax.text(j, i, format(cm[i, j], 'd'),
                                       ha="center", va="center",
                                       color="white" if cm[i, j] > cm.max() / 2. else "black")
                        
                        st.pyplot(fig)
                
                with tab3:
                    st.subheader("Feature Importance")
                    
                    importance = calculate_feature_importance(model, feature_columns, 
                                                            st.session_state.X_encoded.values, 
                                                            st.session_state.y_encoded.values)
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    features = list(importance.keys())
                    values = list(importance.values())
                    
                    bars = ax.bar(features, values, color='skyblue', edgecolor='navy')
                    ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
                    ax.set_ylabel('Importance Score')
                    plt.xticks(rotation=45, ha='right')
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.3f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with tab4:
                    st.subheader("Make Predictions")
                    st.write("Enter values for each feature to get a prediction:")
                    
                    input_values = {}
                    cols = st.columns(min(3, len(feature_columns)))
                    
                    for i, feature in enumerate(feature_columns):
                        with cols[i % len(cols)]:
                            if feature in st.session_state.label_encoders:
                                # Categorical feature
                                options = list(st.session_state.label_encoders[feature].keys())
                                input_values[feature] = st.selectbox(f"{feature}", options, key=f"input_{feature}")
                            else:
                                # Numerical feature
                                min_val = float(st.session_state.X_encoded[feature].min())
                                max_val = float(st.session_state.X_encoded[feature].max())
                                input_values[feature] = st.number_input(f"{feature}", 
                                                                      min_value=min_val, 
                                                                      max_value=max_val, 
                                                                      value=(min_val + max_val) / 2,
                                                                      key=f"input_{feature}")
                    
                    if st.button("🔮 Predict"):
                        # Prepare input for prediction
                        input_array = []
                        for feature in feature_columns:
                            if feature in st.session_state.label_encoders:
                                encoded_val = st.session_state.label_encoders[feature][input_values[feature]]
                                input_array.append(encoded_val)
                            else:
                                input_array.append(input_values[feature])
                        
                        # Make prediction
                        prediction = model.predict(np.array([input_array]))[0]
                        
                        # Decode prediction if target was encoded
                        if hasattr(st.session_state, 'target_encoder'):
                            pred_label = [k for k, v in st.session_state.target_encoder.items() if v == prediction][0]
                            st.success(f"🎯 Prediction: **{pred_label}**")
                        else:
                            st.success(f"🎯 Prediction: **{prediction}**")
    
    else:
        # Sample data section
        st.header("📝 Try with Sample Data")
        if st.button("Load Sample Dataset (Computer Purchase)"):
            sample_data = {
                'age': ['youth', 'youth', 'middle_aged', 'senior', 'senior', 'senior', 
                       'middle_aged', 'youth', 'youth', 'senior', 'youth', 'middle_aged', 
                       'middle_aged', 'senior'],
                'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 
                          'low', 'medium', 'medium', 'medium', 'high', 'medium'],
                'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 
                           'yes', 'no', 'yes', 'no'],
                'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 
                                 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 
                                 'fair', 'excellent'],
                'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 
                                 'yes', 'yes', 'yes', 'yes', 'no']
            }
            
            df_sample = pd.DataFrame(sample_data)
            
            # Save to temporary CSV for download
            csv = df_sample.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample Data",
                data=csv,
                file_name='sample_data.csv',
                mime='text/csv'
            )
            
            st.dataframe(df_sample)
            st.info("💡 Download this sample data and upload it above to try the decision tree builder!")

if __name__ == "__main__":
    main()