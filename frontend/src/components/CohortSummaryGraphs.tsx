import React, { useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { Variable } from '@/types';

interface CohortSummaryGraphsProps {
  variables: { [key: string]: Variable };
}

// Color palettes for the pie charts
const DOMAIN_COLORS = [
  '#3b82f6', // blue
  '#10b981', // green
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // purple
  '#ec4899', // pink
  '#06b6d4', // cyan
  '#f97316', // orange
  '#84cc16', // lime
  '#6366f1', // indigo
];

const TYPE_COLORS = [
  '#0ea5e9', // sky
  '#22c55e', // green
  '#eab308', // yellow
  '#f43f5e', // rose
  '#a855f7', // purple
  '#14b8a6', // teal
  '#fb923c', // orange
  '#4ade80', // green
  '#818cf8', // indigo
  '#fb7185', // rose
];

const CATEGORY_COLORS = [
  '#64748b', // slate - non-categorical
  '#3b82f6', // blue - 2 categories
  '#10b981', // green - 3 categories  
  '#f59e0b', // amber - 4 categories
  '#ef4444', // red - 5+ categories
];

export default function CohortSummaryGraphs({ variables }: CohortSummaryGraphsProps) {
  const [selectedDomain, setSelectedDomain] = useState<string | null>(null);
  const [selectedType, setSelectedType] = useState<string | null>(null);

  // Calculate category distribution
  const categoryDistributionData = useMemo(() => {
    const categoryCounts: { [key: string]: number } = {
      'Non-categorical': 0,
      '2 categories': 0,
      '3 categories': 0,
      '4 categories': 0,
      '5+ categories': 0,
    };
    
    // Filter variables based on selections
    let filteredVars = Object.values(variables);
    if (selectedDomain) {
      filteredVars = filteredVars.filter(v => (v.omop_domain || 'Unmapped') === selectedDomain);
    }
    if (selectedType) {
      filteredVars = filteredVars.filter(v => (v.var_type || 'Unknown') === selectedType);
    }
    
    filteredVars.forEach((variable) => {
      const numCategories = variable.categories?.length || 0;
      
      if (numCategories === 0) {
        categoryCounts['Non-categorical']++;
      } else if (numCategories === 2) {
        categoryCounts['2 categories']++;
      } else if (numCategories === 3) {
        categoryCounts['3 categories']++;
      } else if (numCategories === 4) {
        categoryCounts['4 categories']++;
      } else if (numCategories >= 5) {
        categoryCounts['5+ categories']++;
      }
    });
    
    return Object.entries(categoryCounts).map(([name, value]) => ({ name, value }));
  }, [variables, selectedDomain, selectedType]);

  // Calculate visit types distribution
  const visitTypesData = useMemo(() => {
    const visitCounts: { [key: string]: number } = {};
    
    // Filter variables based on selections
    let filteredVars = Object.values(variables);
    if (selectedDomain) {
      filteredVars = filteredVars.filter(v => (v.omop_domain || 'Unmapped') === selectedDomain);
    }
    if (selectedType) {
      filteredVars = filteredVars.filter(v => (v.var_type || 'Unknown') === selectedType);
    }
    
    filteredVars.forEach((variable) => {
      const visits = variable.visits?.trim();
      
      if (!visits || visits === '' || visits.toLowerCase() === 'not applicable' || visits.toLowerCase() === 'n/a') {
        visitCounts['Not specified'] = (visitCounts['Not specified'] || 0) + 1;
      } else {
        // Split by common delimiters and count each visit type
        const visitTypes = visits.split(/[;,/]/).map(v => v.trim()).filter(v => v.length > 0);
        
        if (visitTypes.length === 0) {
          visitCounts['Not specified'] = (visitCounts['Not specified'] || 0) + 1;
        } else {
          visitTypes.forEach(visitType => {
            visitCounts[visitType] = (visitCounts[visitType] || 0) + 1;
          });
        }
      }
    });
    
    return Object.entries(visitCounts)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value);
  }, [variables, selectedDomain, selectedType]);

  // Calculate OMOP domain distribution
  const domainData = useMemo(() => {
    const domainCounts: { [key: string]: number } = {};
    
    // Filter by selected type if any
    const filteredVars = selectedType 
      ? Object.values(variables).filter(v => (v.var_type || 'Unknown') === selectedType)
      : Object.values(variables);
    
    filteredVars.forEach((variable) => {
      const domain = variable.omop_domain || 'Unmapped';
      domainCounts[domain] = (domainCounts[domain] || 0) + 1;
    });
    
    return Object.entries(domainCounts)
      .map(([name, value]) => ({ 
        name, 
        value,
        itemStyle: { color: DOMAIN_COLORS[Object.keys(domainCounts).indexOf(name) % DOMAIN_COLORS.length] }
      }))
      .sort((a, b) => b.value - a.value);
  }, [variables, selectedType]);

  // Calculate data type distribution
  const typeData = useMemo(() => {
    const typeCounts: { [key: string]: number } = {};
    
    // Filter by selected domain if any
    const filteredVars = selectedDomain
      ? Object.values(variables).filter(v => (v.omop_domain || 'Unmapped') === selectedDomain)
      : Object.values(variables);
    
    filteredVars.forEach((variable) => {
      const type = variable.var_type || 'Unknown';
      typeCounts[type] = (typeCounts[type] || 0) + 1;
    });
    
    return Object.entries(typeCounts)
      .map(([name, value]) => ({ 
        name, 
        value,
        itemStyle: { color: TYPE_COLORS[Object.keys(typeCounts).indexOf(name) % TYPE_COLORS.length] }
      }))
      .sort((a, b) => b.value - a.value);
  }, [variables, selectedDomain]);

  // Domain chart options
  const domainChartOptions = {
    title: {
      text: selectedType ? `Domains for {filter|${selectedType}}` : 'OMOP Domains Distribution',
      left: 'center',
      textStyle: { 
        fontSize: 14, 
        fontWeight: 500,
        rich: {
          filter: {
            color: '#3b82f6',
            fontWeight: 'bold',
            backgroundColor: '#dbeafe',
            padding: [2, 6],
            borderRadius: 4
          }
        }
      }
    },
    tooltip: {
      trigger: 'item',
      formatter: '{b}: {c} ({d}%)'
    },
    legend: {
      orient: 'horizontal',
      bottom: 0,
      type: 'scroll',
      formatter: (name: string) => {
        const item = domainData.find(d => d.name === name);
        return `${name} (${item?.value || 0})`;
      }
    },
    series: [
      {
        type: 'pie',
        radius: ['40%', '70%'],
        avoidLabelOverlap: true,
        itemStyle: {
          borderRadius: 8,
          borderColor: '#fff',
          borderWidth: 2
        },
        label: {
          show: true,
          formatter: '{d}%',
          fontSize: 11
        },
        emphasis: {
          label: {
            show: true,
            fontSize: 14,
            fontWeight: 'bold'
          },
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        },
        data: domainData
      }
    ]
  };

  // Type chart options
  const typeChartOptions = {
    title: {
      text: selectedDomain ? `Types in {filter|${selectedDomain}}` : 'Data Types Distribution',
      left: 'center',
      textStyle: { 
        fontSize: 14, 
        fontWeight: 500,
        rich: {
          filter: {
            color: '#10b981',
            fontWeight: 'bold',
            backgroundColor: '#d1fae5',
            padding: [2, 6],
            borderRadius: 4
          }
        }
      }
    },
    tooltip: {
      trigger: 'item',
      formatter: '{b}: {c} ({d}%)'
    },
    legend: {
      orient: 'horizontal',
      bottom: 0,
      type: 'scroll',
      formatter: (name: string) => {
        const item = typeData.find(d => d.name === name);
        return `${name} (${item?.value || 0})`;
      }
    },
    series: [
      {
        type: 'pie',
        radius: ['40%', '70%'],
        avoidLabelOverlap: true,
        itemStyle: {
          borderRadius: 8,
          borderColor: '#fff',
          borderWidth: 2
        },
        label: {
          show: true,
          formatter: '{d}%',
          fontSize: 11
        },
        emphasis: {
          label: {
            show: true,
            fontSize: 14,
            fontWeight: 'bold'
          },
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        },
        data: typeData
      }
    ]
  };

  // Category distribution chart options
  const categoryChartOptions = {
    title: {
      text: selectedDomain || selectedType 
        ? `Categorization${selectedDomain ? ` in {filter|${selectedDomain}}` : ''}${selectedType ? ` for {filter|${selectedType}}` : ''}`
        : 'Variable Categorization Distribution',
      left: 'center',
      textStyle: { 
        fontSize: 14, 
        fontWeight: 500,
        rich: {
          filter: {
            color: '#f59e0b',
            fontWeight: 'bold',
            backgroundColor: '#fef3c7',
            padding: [2, 6],
            borderRadius: 4
          }
        }
      }
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      },
      formatter: (params: any) => {
        const data = params[0];
        const total = categoryDistributionData.reduce((sum, item) => sum + item.value, 0);
        const percentage = total > 0 ? ((data.value / total) * 100).toFixed(1) : '0';
        return `${data.name}<br/>Count: ${data.value} (${percentage}%)`;
      }
    },
    xAxis: {
      type: 'category',
      data: categoryDistributionData.map(d => d.name),
      axisLabel: {
        interval: 0,
        rotate: 0,
        fontSize: 11
      }
    },
    yAxis: {
      type: 'value',
      name: 'Number of Variables',
      nameTextStyle: {
        fontSize: 12
      }
    },
    series: [
      {
        type: 'bar',
        data: categoryDistributionData.map((d, index) => ({
          value: d.value,
          itemStyle: {
            color: CATEGORY_COLORS[index],
            borderRadius: [8, 8, 0, 0]
          }
        })),
        barWidth: '60%',
        label: {
          show: true,
          position: 'top',
          fontSize: 11,
          fontWeight: 'bold'
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }
    ],
    grid: {
      left: '12%',
      right: '8%',
      bottom: '10%',
      top: '20%'
    }
  };

  // Visit types distribution chart options
  const visitTypesChartOptions = {
    title: {
      text: selectedDomain || selectedType 
        ? `Visit Types${selectedDomain ? ` in {filter|${selectedDomain}}` : ''}${selectedType ? ` for {filter|${selectedType}}` : ''}`
        : 'Visit Types Distribution',
      left: 'center',
      textStyle: { 
        fontSize: 14, 
        fontWeight: 500,
        rich: {
          filter: {
            color: '#8b5cf6',
            fontWeight: 'bold',
            backgroundColor: '#ede9fe',
            padding: [2, 6],
            borderRadius: 4
          }
        }
      }
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      },
      formatter: (params: any) => {
        const data = params[0];
        const total = visitTypesData.reduce((sum, item) => sum + item.value, 0);
        const percentage = total > 0 ? ((data.value / total) * 100).toFixed(1) : '0';
        return `${data.name}<br/>Count: ${data.value} (${percentage}%)`;
      }
    },
    xAxis: {
      type: 'category',
      data: visitTypesData.map(d => d.name),
      axisLabel: {
        interval: 0,
        rotate: 45,
        fontSize: 10
      }
    },
    yAxis: {
      type: 'value',
      name: 'Number of Variables',
      nameTextStyle: {
        fontSize: 12
      }
    },
    series: [
      {
        type: 'bar',
        data: visitTypesData.map((d) => ({
          value: d.value,
          itemStyle: {
            color: '#8b5cf6', // purple
            borderRadius: [8, 8, 0, 0]
          }
        })),
        barWidth: '60%',
        label: {
          show: true,
          position: 'top',
          fontSize: 11,
          fontWeight: 'bold'
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }
    ],
    grid: {
      left: '12%',
      right: '8%',
      bottom: '30%',
      top: '20%'
    }
  };

  // Handle domain chart click
  const onDomainClick = (params: any) => {
    if (params.name === selectedDomain) {
      setSelectedDomain(null); // Deselect if clicking same domain
    } else {
      setSelectedDomain(params.name);
    }
  };

  // Handle type chart click
  const onTypeClick = (params: any) => {
    if (params.name === selectedType) {
      setSelectedType(null); // Deselect if clicking same type
    } else {
      setSelectedType(params.name);
    }
  };

  if (Object.keys(variables).length === 0) {
    return (
      <div className="bg-white shadow-md rounded-lg p-4 mb-4">
        <h3 className="text-lg font-semibold mb-3 border-b pb-2">Summary Graphs</h3>
        <p className="text-gray-500"><em>No variables available to display graphs</em></p>
      </div>
    );
  }

  return (
    <div className="bg-white shadow-md rounded-lg p-4 mb-4">
      <div className="flex justify-between items-center mb-3 border-b pb-2">
        <h3 className="text-lg font-semibold">Summary Graphs</h3>
        {(selectedDomain || selectedType) && (
          <button
            onClick={() => {
              setSelectedDomain(null);
              setSelectedType(null);
            }}
            className="text-xs bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded-full transition-colors"
          >
            Reset Filters
          </button>
        )}
      </div>
      
      {(selectedDomain || selectedType) && (
        <div className="mb-3 text-sm text-gray-600 bg-blue-50 px-3 py-2 rounded">
          <strong>💡 Tip:</strong> Charts are linked! 
          {selectedDomain && ` Filtering by domain: "${selectedDomain}"`}
          {selectedType && ` Filtering by type: "${selectedType}"`}
          {' '}Click a slice again to deselect.
        </div>
      )}
      
      <div className="space-y-6">
        {/* Top row: Pie charts */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* OMOP Domains Pie Chart */}
          <div className="cursor-pointer">
            <ReactECharts
              option={domainChartOptions}
              style={{ height: '350px' }}
              onEvents={{
                click: onDomainClick
              }}
            />
          </div>

          {/* Data Types Pie Chart */}
          <div className="cursor-pointer">
            <ReactECharts
              option={typeChartOptions}
              style={{ height: '350px' }}
              onEvents={{
                click: onTypeClick
              }}
            />
          </div>
        </div>

        {/* Bottom row: Bar charts */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Category distribution bar chart */}
          <div>
            <ReactECharts
              option={categoryChartOptions}
              style={{ height: '350px' }}
            />
          </div>

          {/* Visit types distribution bar chart */}
          <div>
            <ReactECharts
              option={visitTypesChartOptions}
              style={{ height: '350px' }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
