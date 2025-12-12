import React, { useMemo, useState, useEffect, useRef } from 'react';
import ReactECharts from 'echarts-for-react';
import { Variable } from '@/types';

interface CohortSummaryGraphsProps {
  variables: { [key: string]: Variable };
  isExpanded?: boolean;
  cohortId: string;
  selectedOMOPDomains: Set<string>;
  selectedDataTypes: Set<string>;
  selectedCategoryTypes: Set<string>;
  selectedVisitTypes: Set<string>;
  onDomainClick: (domain: string | null) => void;
  onTypeClick: (type: string | null) => void;
  onCategoryClick: (category: string | null) => void;
  onVisitTypeClick: (visitType: string | null) => void;
  onResetFilters: () => void;
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
  '#6366f1', // indigo - all categorical
  '#3b82f6', // blue - 2 categories
  '#10b981', // green - 3 categories  
  '#f59e0b', // amber - 4+ categories
];

const CohortSummaryGraphs = React.memo(function CohortSummaryGraphs({ 
  variables, 
  isExpanded = false,
  cohortId,
  selectedOMOPDomains,
  selectedDataTypes,
  selectedCategoryTypes,
  selectedVisitTypes,
  onDomainClick,
  onTypeClick,
  onCategoryClick,
  onVisitTypeClick,
  onResetFilters
}: CohortSummaryGraphsProps) {
  const [shouldRender, setShouldRender] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Convert Sets to single selected values for chart highlighting (use first item or null)
  const selectedDomain = selectedOMOPDomains.size > 0 ? Array.from(selectedOMOPDomains)[0] : null;
  const selectedType = selectedDataTypes.size > 0 ? Array.from(selectedDataTypes)[0] : null;
  const selectedCategory = selectedCategoryTypes.size > 0 ? Array.from(selectedCategoryTypes)[0] : null;
  const selectedVisitType = selectedVisitTypes.size > 0 ? Array.from(selectedVisitTypes)[0] : null;

  // Use Intersection Observer to render only when scrolled into view
  useEffect(() => {
    // Only start observing if cohort is expanded
    if (!isExpanded) {
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !shouldRender) {
            setShouldRender(true);
          }
        });
      },
      {
        rootMargin: '200px', // Start loading before it comes into view
        threshold: 0.1,
      }
    );

    const currentContainer = containerRef.current;
    if (currentContainer) {
      observer.observe(currentContainer);
    }

    return () => {
      if (currentContainer) {
        observer.unobserve(currentContainer);
      }
    };
  }, [isExpanded, shouldRender]);

  // Helper function to apply all filters to variables - create this ONCE and reuse
  const getFilteredVariables = useMemo(() => {
    if (!shouldRender) return [];
    
    return Object.values(variables).filter((v) => {
      // Apply domain filter
      if (selectedDomain && (v.omop_domain || 'Unmapped') !== selectedDomain) {
        return false;
      }
      
      // Apply type filter
      if (selectedType && (v.var_type || 'Unknown') !== selectedType) {
        return false;
      }
      
      // Apply category filter
      if (selectedCategory) {
        const numCategories = v.categories?.length || 0;
        if (selectedCategory === 'Non-categorical' && numCategories !== 0) return false;
        if (selectedCategory === 'All categorical' && numCategories === 0) return false;
        if (selectedCategory === '2 categories' && numCategories !== 2) return false;
        if (selectedCategory === '3 categories' && numCategories !== 3) return false;
        if (selectedCategory === '4+ categories' && numCategories < 4) return false;
      }
      
      // Apply visit type filter
      if (selectedVisitType) {
        const visits = v.visits?.trim();
        if (selectedVisitType === 'Not specified') {
          if (visits && visits !== '' && visits.toLowerCase() !== 'not applicable' && visits.toLowerCase() !== 'n/a') {
            return false;
          }
        } else if (!visits?.includes(selectedVisitType)) {
          return false;
        }
      }
      
      return true;
    });
  }, [variables, selectedDomain, selectedType, selectedCategory, selectedVisitType, shouldRender]);

  // Calculate category distribution - ONLY when shouldRender is true
  const categoryDistributionData = useMemo(() => {
    if (!shouldRender) return [];
    
    const categoryCounts: { [key: string]: number } = {
      'Non-categorical': 0,
      'All categorical': 0,
      '2 categories': 0,
      '3 categories': 0,
      '4+ categories': 0,
    };
    
    // Use pre-filtered variables (filter out category filter for this chart)
    const filteredVars = selectedCategory 
      ? Object.values(variables).filter((v) => {
          if (selectedDomain && (v.omop_domain || 'Unmapped') !== selectedDomain) return false;
          if (selectedType && (v.var_type || 'Unknown') !== selectedType) return false;
          if (selectedVisitType) {
            const visits = v.visits?.trim();
            if (selectedVisitType === 'Not specified') {
              if (visits && visits !== '' && visits.toLowerCase() !== 'not applicable' && visits.toLowerCase() !== 'n/a') return false;
            } else if (!visits?.includes(selectedVisitType)) return false;
          }
          return true;
        })
      : getFilteredVariables;
    
    filteredVars.forEach((variable) => {
      const numCategories = variable.categories?.length || 0;
      
      if (numCategories === 0) {
        categoryCounts['Non-categorical']++;
      } else {
        categoryCounts['All categorical']++;
        if (numCategories === 2) {
          categoryCounts['2 categories']++;
        } else if (numCategories === 3) {
          categoryCounts['3 categories']++;
        } else if (numCategories >= 4) {
          categoryCounts['4+ categories']++;
        }
      }
    });
    
    return Object.entries(categoryCounts).map(([name, value]) => ({ name, value }));
  }, [variables, selectedDomain, selectedType, selectedCategory, selectedVisitType, getFilteredVariables, shouldRender]);

  // Calculate visit types distribution - ONLY when shouldRender is true
  const visitTypesData = useMemo(() => {
    if (!shouldRender) return [];
    
    const visitCounts: { [key: string]: number } = {};
    
    // Use pre-filtered variables (filter out visit type filter for this chart)
    const filteredVars = selectedVisitType
      ? Object.values(variables).filter((v) => {
          if (selectedDomain && (v.omop_domain || 'Unmapped') !== selectedDomain) return false;
          if (selectedType && (v.var_type || 'Unknown') !== selectedType) return false;
          if (selectedCategory) {
            const numCategories = v.categories?.length || 0;
            if (selectedCategory === 'Non-categorical' && numCategories !== 0) return false;
            if (selectedCategory === '2 categories' && numCategories !== 2) return false;
            if (selectedCategory === '3 categories' && numCategories !== 3) return false;
            if (selectedCategory === '4 categories' && numCategories !== 4) return false;
            if (selectedCategory === '5+ categories' && numCategories < 5) return false;
          }
          return true;
        })
      : getFilteredVariables;
    
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
  }, [variables, selectedDomain, selectedType, selectedCategory, selectedVisitType, getFilteredVariables, shouldRender]);

  // Calculate OMOP domain distribution - ONLY when shouldRender is true
  const domainData = useMemo(() => {
    if (!shouldRender) return [];
    
    const domainCounts: { [key: string]: number } = {};
    
    // Use pre-filtered variables (filter out domain filter for this chart)
    const filteredVars = selectedDomain
      ? Object.values(variables).filter((v) => {
          if (selectedType && (v.var_type || 'Unknown') !== selectedType) return false;
          if (selectedCategory) {
            const numCategories = v.categories?.length || 0;
            if (selectedCategory === 'Non-categorical' && numCategories !== 0) return false;
            if (selectedCategory === '2 categories' && numCategories !== 2) return false;
            if (selectedCategory === '3 categories' && numCategories !== 3) return false;
            if (selectedCategory === '4 categories' && numCategories !== 4) return false;
            if (selectedCategory === '5+ categories' && numCategories < 5) return false;
          }
          if (selectedVisitType) {
            const visits = v.visits?.trim();
            if (selectedVisitType === 'Not specified') {
              if (visits && visits !== '' && visits.toLowerCase() !== 'not applicable' && visits.toLowerCase() !== 'n/a') return false;
            } else if (!visits?.includes(selectedVisitType)) return false;
          }
          return true;
        })
      : getFilteredVariables;
    
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
  }, [variables, selectedDomain, selectedType, selectedCategory, selectedVisitType, getFilteredVariables, shouldRender]);

  // Calculate data type distribution - ONLY when shouldRender is true
  const typeData = useMemo(() => {
    if (!shouldRender) return [];
    
    const typeCounts: { [key: string]: number } = {};
    
    // Use pre-filtered variables (filter out type filter for this chart)
    const filteredVars = selectedType
      ? Object.values(variables).filter((v) => {
          if (selectedDomain && (v.omop_domain || 'Unmapped') !== selectedDomain) return false;
          if (selectedCategory) {
            const numCategories = v.categories?.length || 0;
            if (selectedCategory === 'Non-categorical' && numCategories !== 0) return false;
            if (selectedCategory === '2 categories' && numCategories !== 2) return false;
            if (selectedCategory === '3 categories' && numCategories !== 3) return false;
            if (selectedCategory === '4 categories' && numCategories !== 4) return false;
            if (selectedCategory === '5+ categories' && numCategories < 5) return false;
          }
          if (selectedVisitType) {
            const visits = v.visits?.trim();
            if (selectedVisitType === 'Not specified') {
              if (visits && visits !== '' && visits.toLowerCase() !== 'not applicable' && visits.toLowerCase() !== 'n/a') return false;
            } else if (!visits?.includes(selectedVisitType)) return false;
          }
          return true;
        })
      : getFilteredVariables;
    
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
  }, [variables, selectedDomain, selectedType, selectedCategory, selectedVisitType, getFilteredVariables, shouldRender]);

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
          formatter: '{b}\n{d}%',
          fontSize: 10,
          fontWeight: 'bold'
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
          formatter: '{b}\n{d}%',
          fontSize: 10,
          fontWeight: 'bold'
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

  // Handle chart clicks - these are internal handlers that call the parent callbacks
  const handleDomainChartClick = (params: any) => {
    onDomainClick(params.name);
  };

  const handleTypeChartClick = (params: any) => {
    onTypeClick(params.name);
  };

  const handleCategoryChartClick = (params: any) => {
    onCategoryClick(params.name);
  };

  const handleVisitTypeChartClick = (params: any) => {
    onVisitTypeClick(params.name);
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
          <div className="animate-spin-slow rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
          <p className="text-gray-600 font-medium">Rendering charts...</p>
          <p className="text-gray-400 text-sm mt-2">This will take just a moment</p>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="bg-white shadow-md rounded-lg p-4 mb-4">
      <div className="mb-3 border-b pb-2">
        <h3 className="text-lg font-semibold">Summary Graphs</h3>
      </div>
      
      {/* Interactive charts note - always visible */}
      <div className="mb-3 text-sm text-gray-600 bg-blue-50 px-3 py-2 rounded">
        <strong>💡 NOTE:</strong> All charts are interactive. Click on a value to re-generate the graphs for that value. Click again to deselect, or use the &quot;Reset Filters&quot; button.
      </div>
      
      <div className="space-y-6">
        {/* Top row: Pie charts */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* OMOP Domains Pie Chart */}
          <div className="cursor-pointer">
            <ReactECharts
              option={domainChartOptions}
              style={{ height: '350px' }}
              onEvents={{
                click: handleDomainChartClick
              }}
            />
          </div>

          {/* Data Types Pie Chart */}
          <div className="cursor-pointer">
            <ReactECharts
              option={typeChartOptions}
              style={{ height: '350px' }}
              onEvents={{
                click: handleTypeChartClick
              }}
            />
          </div>
        </div>

        {/* Bottom row: Bar charts */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-12">
          {/* Category distribution bar chart */}
          <div className="cursor-pointer">
            <ReactECharts
              option={categoryChartOptions}
              style={{ height: '350px' }}
              onEvents={{
                click: handleCategoryChartClick
              }}
            />
          </div>

          {/* Visit types distribution bar chart */}
          <div className="cursor-pointer">
            <ReactECharts
              option={visitTypesChartOptions}
              style={{ height: '350px' }}
              onEvents={{
                click: handleVisitTypeChartClick
              }}
            />
          </div>
        </div>
      </div>
      
      {/* Filter summary and Reset button at bottom - always visible */}
      <div className="mt-6 pt-4 border-t">
        <div className="text-sm text-gray-600 mb-3 text-center">
          <span className="font-semibold">Filters in effect:</span>
          {!(selectedDomain || selectedType || selectedCategory || selectedVisitType) && ' None'}
          {selectedDomain && ` OMOP Domain = ${selectedDomain}`}
          {selectedType && `${selectedDomain ? ',' : ''} Data Type = ${selectedType}`}
          {selectedCategory && `${selectedDomain || selectedType ? ',' : ''} Category = ${selectedCategory}`}
          {selectedVisitType && `${selectedDomain || selectedType || selectedCategory ? ',' : ''} Visit Type = ${selectedVisitType}`}
        </div>
        {(selectedDomain || selectedType || selectedCategory || selectedVisitType) && (
          <div className="flex justify-center">
            <button
              onClick={onResetFilters}
              className="bg-blue-600 hover:bg-blue-700 text-white font-semibold px-6 py-2.5 rounded-lg shadow-md hover:shadow-lg transition-all duration-200 flex items-center gap-2"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
              </svg>
              Reset All Filters
            </button>
          </div>
        )}
      </div>
    </div>
  );
});

export default CohortSummaryGraphs;
