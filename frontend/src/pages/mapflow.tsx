'use client';

import React, {useState, useEffect, useCallback} from 'react';
import {useRouter} from 'next/router';
import ReactFlow, {
  Node,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Edge,
  ConnectionLineType,
  MiniMap
} from 'reactflow';
import FlowNode from '../components/FlowNode';

import 'reactflow/dist/style.css';

const initialNodes: Node[] = [];
const initialEdges: Edge[] = [];

const nodeTypes = {
  custom: FlowNode
};

const defaultEdgeOptions = {
  animated: true,
  type: 'smoothstep'
};

export default function MapFlow() {
  const [selectedFile, setSelectedFile] = useState('patient_register_UMV1');

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const onConnect = useCallback((params: Connection | Edge) => setEdges(eds => addEdge(params, eds)), [setEdges]);

  const prepareNodes = (dataDict: any, filename: string) => {
    let y = 5;
    const prepNodes = [];
    if (dataDict[filename]) {
      for (const variable of dataDict[filename]) {
        prepNodes.push({
          id: variable.Name,
          type: 'custom',
          data: {
            label: variable.Name,
            description: variable.Description
          },
          position: {x: 10, y: y},
          className: 'flowNode'
        });
        y += 110;
      }
      setNodes(prepNodes);
    }
  };

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL;
    fetch(`${apiUrl}/summary`, {
      credentials: 'include'
      // headers: {
      //   'Authorization': `Bearer ${token}`
      // }
    })
      .then(response => response.json())
      .then(data => {
        // setDataDict(data)
        console.log('dataDict', data);
        prepareNodes(data, selectedFile);
        // TODO: generate nodes from data dict
      });
  }, []);

  return (
    <main className="w-full p-4">
      <div className="flow h-screen w-full">
        <ReactFlow
          nodes={nodes}
          onNodesChange={onNodesChange}
          edges={edges}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          defaultEdgeOptions={defaultEdgeOptions}
          connectionLineType={ConnectionLineType.SmoothStep}
          fitView
          panOnScroll
        >
          <MiniMap nodeStrokeWidth={3} pannable zoomable />
        </ReactFlow>
      </div>
    </main>
  );
}
