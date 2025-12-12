import streamlit as st
from yfiles_graphs_for_streamlit import StreamlitGraphWidget, Node, Edge, EdgeStyle, DashStyle, Layout, LabelStyle

def app():
    st.title("yFiles Directed Test")
    st.info("Interactive playground to understand the `directed` argument.")

    # 1. Controls
    st.sidebar.header("Controls")
    # This global checkbox allows you to toggle the default, but our custom styles will override it
    global_directed = st.sidebar.checkbox("Global Defaults: widget.directed", value=False)
    
    # 2. Define Nodes
    st.subheader("Graph Definitions")
    c1, c2, c3 = st.columns(3)
    c1.metric("A -> B", "Solid / Directed")
    c2.metric("B - C", "Dashed / Undirected")
    c3.metric("C -> A", "Dotted / Directed")

    nodes = [
        Node("A", properties={"label": "A", "color": "red", "shape": "rectangle"}),
        Node("B", properties={"label": "B", "color": "blue", "shape": "ellipse"}),
        Node("C", properties={"label": "C", "color": "green", "shape": "hexagon"})
    ]

    # 3. Define Edges with visual properties
    edges = [
        # A -> B: Solid, Black, Directed
        Edge("A", "B", properties={
            "label": "A->B", 
            "color": "black", 
            "dash": "solid", 
            "directed": True
        }),
        
        # B -> C: Dashed, Gray, Undirected (No Arrow)
        Edge("B", "C", properties={
            "label": "B-C", 
            "color": "gray", 
            "dash": "dashed", 
            "directed": False
        }),
        
        # C -> A: Dotted, Orange, Directed
        Edge("C", "A", properties={
            "label": "C->A", 
            "color": "orange", 
            "dash": "dotted", 
            "directed": True
        })
    ]

    # 4. Style Logic Function
    def get_edge_style(edge):
        props = edge['properties']
        
        # Handle Dash Styles
        d_str = props.get('dash', 'solid')
        if d_str == 'dashed': 
            d_style = DashStyle.DASH
        elif d_str == 'dotted': 
            d_style = DashStyle.DOT
        else: 
            d_style = DashStyle.SOLID
        
        # Handle Direction (Arrow)
        # This boolean tells yFiles to render the arrowhead
        is_directed = props.get('directed', True)

        return EdgeStyle(
            color=props.get('color', 'black'),
            dash_style=d_style,
            directed=is_directed,
            thickness=2.0
        )

    # 5. Initialize Widget
    st.subheader("Rendered Graph")
    widget = StreamlitGraphWidget(nodes=nodes, edges=edges)
    
    # 6. Apply Mappings
    # Set the global default first
    widget.directed = global_directed 
    
    # Map Node Properties
    widget.node_label_mapping = lambda n: n['properties']['label']
    widget.node_color_mapping = lambda n: n['properties']['color']
    # widget.node_shape_mapping can be added if your version supports it, 
    # otherwise shapes default to rectangle/ellipse based on layout.

    # Map Edge Styles (Crucial for arrows/dashes)
    widget.edge_label_mapping = lambda e: e['properties']['label']
    widget.edge_styles_mapping = get_edge_style

    # 7. Render
    # Note: height is set as a property, not in show()
    widget.height = 600
    selected = widget.show(key="yfiles_test_graph", graph_layout=Layout.HIERARCHIC)
    
    if selected:
        st.success(f"Selected Node: {selected}")

if __name__ == "__main__":
    app()