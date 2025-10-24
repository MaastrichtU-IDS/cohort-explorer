import React, { useMemo, useState, useEffect, useRef } from 'react';
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
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [selectedVisitType, setSelectedVisitType] = useState<string | null>(null);
  const [isVisible, setIsVisible] = useState(false);
  const [shouldRender, setShouldRender] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Intersection Observer for lazy loading
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !isVisible) {
            setIsVisible(true);
            // Wait 500ms before rendering to avoid rendering during fast scrolling
            const timer = setTimeout(() => {
              setShouldRender(true);
            }, 500);
            return () => clearTimeout(timer);
          }
        });
      },
      {
        rootMargin: '100px', // Start loading slightly before it comes into view
        threshold: 0.1,
      }
    );

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => {
      if (containerRef.current) {
        observer.unobserve(containerRef.current);
      }
    };
  }, [isVisible]);

  // Calculate category distribution - ONLY when shouldRender is true
  const categoryDistributionData = useMemo(() => {
    if (!shouldRender) return [];
    
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
  }, [variables, selectedDomain, selectedType, shouldRender]);

  // Calculate visit types distribution - ONLY when shouldRender is true
  const visitTypesData = useMemo(() => {
    if (!shouldRender) return [];
    
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
  }, [variables, selectedDomain, selectedType, shouldRender]);

  // Calculate OMOP domain distribution - ONLY when shouldRender is true
  const domainData = useMemo(() => {
    if (!shouldRender) return [];
    
    const domainCounts: { [key: string]: number } = {};
    
    // Apply all filters
    let filteredVars = Object.values(variables);
    if (selectedType) {
      filteredVars = filteredVars.filter(v => (v.var_type || 'Unknown') === selectedType);
    }
    if (selectedCategory) {
      filteredVars = filteredVars.filter(v => {
        const numCategories = v.categories?.length || 0;
        if (selectedCategory === 'Non-categorical') return numCategories === 0;
        if (selectedCategory === '2 categories') return numCategories === 2;
        if (selectedCategory === '3 categories') return numCategories === 3;
        if (selectedCategory === '4 categories') return numCategories === 4;
        if (selectedCategory === '5+ categories') return numCategories >= 5;
        return false;
      });
    }
    if (selectedVisitType) {
      filteredVars = filteredVars.filter(v => {
        const visits = v.visits?.trim();
        if (selectedVisitType === 'Not specified') {
          return !visits || visits === '' || visits.toLowerCase() === 'not applicable' || visits.toLowerCase() === 'n/a';
        }
        return visits?.includes(selectedVisitType);
      });
    }
    
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
  }, [variables, selectedType, selectedCategory, selectedVisitType, shouldRender]);

  // Calculate data type distribution - ONLY when shouldRender is true
  const typeData = useMemo(() => {
    if (!shouldRender) return [];
    
    const typeCounts: { [key: string]: number } = {};
    
    // Apply all filters
    let filteredVars = Object.values(variables);
    if (selectedDomain) {
      filteredVars = filteredVars.filter(v => (v.omop_domain || 'Unmapped') === selectedDomain);
    }
    if (selectedCategory) {
      filteredVars = filteredVars.filter(v => {
        const numCategories = v.categories?.length || 0;
        if (selectedCategory === 'Non-categorical') return numCategories === 0;
        if (selectedCategory === '2 categories') return numCategories === 2;
        if (selectedCategory === '3 categories') return numCategories === 3;
        if (selectedCategory === '4 categories') return numCategories === 4;
        if (selectedCategory === '5+ categories') return numCategories >= 5;
        return false;
      });
    }
    if (selectedVisitType) {
      filteredVars = filteredVars.filter(v => {
        const visits = v.visits?.trim();
        if (selectedVisitType === 'Not specified') {
          return !visits || visits === '' || visits.toLowerCase() === 'not applicable' || visits.toLowerCase() === 'n/a';
        }
        return visits?.includes(selectedVisitType);
      });
    }
    
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
  }, [variables, selectedDomain, selectedCategory, selectedVisitType, shouldRender]);

  // Domain chart options
  const domainChartOptions = {
    title: {
      text: selectedType || selectedCategory || selectedVisitType
        ? `Domains${selectedType ? ` for {filter|${selectedType}}` : ''}${selectedCategory ? ` with {filter|${selectedCategory}}` : ''}${selectedVisitType ? ` in {filter|${selectedVisitType}}` : ''}`
        : 'OMOP Domains Distribution',
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
      text: selectedDomain || selectedCategory || selectedVisitType
        ? `Types${selectedDomain ? ` in {filter|${selectedDomain}}` : ''}${selectedCategory ? ` with {filter|${selectedCategory}}` : ''}${selectedVisitType ? ` for {filter|${selectedVisitType}}` : ''}`
        : 'Data Types Distribution',
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
      text: selectedDomain || selectedType || selectedVisitType
        ? `Categorization${selectedDomain ? ` in {filter|${selectedDomain}}` : ''}${selectedType ? ` for {filter|${selectedType}}` : ''}${selectedVisitType ? ` with {filter|${selectedVisitType}}` : ''}`
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
      text: selectedDomain || selectedType || selectedCategory
        ? `Visit Types${selectedDomain ? ` in {filter|${selectedDomain}}` : ''}${selectedType ? ` for {filter|${selectedType}}` : ''}${selectedCategory ? ` with {filter|${selectedCategory}}` : ''}`
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

  // Handle category chart click
  const onCategoryClick = (params: any) => {
    if (params.name === selectedCategory) {
      setSelectedCategory(null); // Deselect if clicking same category
    } else {
      setSelectedCategory(params.name);
    }
  };

  // Handle visit type chart click
  const onVisitTypeClick = (params: any) => {
    if (params.name === selectedVisitType) {
      setSelectedVisitType(null); // Deselect if clicking same visit type
    } else {
      setSelectedVisitType(params.name);
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

  // Show placeholder while waiting to render
  if (!shouldRender) {
    return (
      <div ref={containerRef} className="bg-white shadow-md rounded-lg p-4 mb-4">
        <div className="mb-3 border-b pb-2">
          <h3 className="text-lg font-semibold">Summary Graphs</h3>
        </div>
        <div className="flex flex-col items-center justify-center py-16">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
          <p className="text-gray-600 font-medium">Rendering variable graphs...</p>
          <p className="text-gray-400 text-sm mt-2">Please wait a moment</p>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="bg-white shadow-md rounded-lg p-4 mb-4">
      <div className="mb-3 border-b pb-2">
        <h3 className="text-lg font-semibold">Summary Graphs</h3>
      </div>
      
      {(selectedDomain || selectedType || selectedCategory || selectedVisitType) && (
        <div className="mb-3 text-sm text-gray-600 bg-blue-50 px-3 py-2 rounded">
          <strong>💡 Tip:</strong> All charts are interactive! 
          {selectedDomain && ` Filtering by domain: "${selectedDomain}".`}
          {selectedType && ` Filtering by type: "${selectedType}".`}
          {selectedCategory && ` Filtering by category: "${selectedCategory}".`}
          {selectedVisitType && ` Filtering by visit type: "${selectedVisitType}".`}
          {' '}Click again to deselect.
        </div>
      )}
      
      {(selectedDomain || selectedType || selectedCategory || selectedVisitType) && (
        <div className="mb-4 flex justify-center">
          <button
            onClick={() => {
              setSelectedDomain(null);
              setSelectedType(null);
              setSelectedCategory(null);
              setSelectedVisitType(null);
            }}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold px-6 py-2.5 rounded-lg shadow-md hover:shadow-lg transition-all duration-200 flex items-center gap-2"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
            </svg>
            Reset All Filters
          </button>
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
          <div className="cursor-pointer">
            <ReactECharts
              option={categoryChartOptions}
              style={{ height: '350px' }}
              onEvents={{
                click: onCategoryClick
              }}
            />
          </div>

          {/* Visit types distribution bar chart */}
          <div className="cursor-pointer">
            <ReactECharts
              option={visitTypesChartOptions}
              style={{ height: '350px' }}
              onEvents={{
                click: onVisitTypeClick
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
