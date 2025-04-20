"use client";

import React, { useEffect, useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  Filler,
  ChartOptions,
} from "chart.js";
import { Line, Bar, Doughnut } from "react-chartjs-2";
import { useTheme } from "next-themes";
import { cn } from "@/helpers/common.helper";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export type ChartType = "line" | "bar" | "doughnut";

interface DashboardChartProps {
  type?: ChartType;
  title: string;
  data: any;
  height?: number;
  className?: string;
  showLegend?: boolean;
}

export const DashboardChart = ({
  type = "line",
  title,
  data,
  height = 300,
  className,
  showLegend = true,
}: DashboardChartProps) => {
  const { resolvedTheme } = useTheme();
  const [chartData, setChartData] = useState(data);
  const [chartOptions, setChartOptions] = useState<ChartOptions<any>>({});

  // Update chart styles based on theme
  useEffect(() => {
    const isDark = resolvedTheme === "dark";

    // Common chart options
    const baseOptions: ChartOptions<any> = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: showLegend,
          position: "top" as const,
          align: "end" as const,
          labels: {
            boxWidth: 12,
            boxHeight: 12,
            padding: 16,
            usePointStyle: true,
            font: {
              size: 11,
              family: "'Inter', sans-serif",
            },
            color: isDark ? "#d1d5db" : "#4b5563",
          },
        },
        title: {
          display: false,
        },
        tooltip: {
          backgroundColor: isDark ? "#1f2937" : "#ffffff",
          titleColor: isDark ? "#f9fafb" : "#1f2937",
          bodyColor: isDark ? "#d1d5db" : "#4b5563",
          borderColor: isDark ? "#374151" : "#e5e7eb",
          borderWidth: 1,
          padding: 12,
          cornerRadius: 8,
          boxPadding: 4,
          usePointStyle: true,
          titleFont: {
            family: "'Inter', sans-serif",
            weight: "600",
          },
          bodyFont: {
            family: "'Inter', sans-serif",
          },
          boxWidth: 8,
        },
      },
    };

    // Chart-specific options
    const typeSpecificOptions: ChartOptions<any> = {};

    if (type === "line") {
      typeSpecificOptions.scales = {
        x: {
          grid: {
            display: false,
            color: isDark ? "#374151" : "#e5e7eb",
          },
          ticks: {
            font: {
              size: 11,
              family: "'Inter', sans-serif",
            },
            color: isDark ? "#9ca3af" : "#6b7280",
          },
        },
        y: {
          grid: {
            color: isDark ? "#374151" : "#e5e7eb",
          },
          ticks: {
            font: {
              size: 11,
              family: "'Inter', sans-serif",
            },
            color: isDark ? "#9ca3af" : "#6b7280",
            padding: 8,
          },
          border: {
            dash: [5, 5],
          },
        },
      };

      // Apply theme-specific styling to the datasets
      if (chartData?.datasets) {
        const updatedDatasets = chartData.datasets.map((dataset: any, index: number) => {
          // Gradient fill for first dataset
          if (index === 0) {
            return {
              ...dataset,
              borderColor: isDark ? "rgb(79, 70, 229)" : "rgb(99, 102, 241)",
              tension: 0.3,
              pointBackgroundColor: isDark ? "rgb(79, 70, 229)" : "rgb(99, 102, 241)",
              pointBorderColor: isDark ? "#1f2937" : "#ffffff",
              pointBorderWidth: 2,
              pointRadius: 4,
              pointHoverRadius: 6,
              fill: true,
              backgroundColor: (context: any) => {
                const ctx = context.chart.ctx;
                const gradient = ctx.createLinearGradient(0, 0, 0, context.chart.height);
                gradient.addColorStop(0, isDark ? "rgba(79, 70, 229, 0.3)" : "rgba(99, 102, 241, 0.2)");
                gradient.addColorStop(1, isDark ? "rgba(79, 70, 229, 0.02)" : "rgba(99, 102, 241, 0.05)");
                return gradient;
              },
            };
          }
          return dataset;
        });

        setChartData({
          ...chartData,
          datasets: updatedDatasets,
        });
      }
    }

    if (type === "bar") {
      typeSpecificOptions.scales = {
        x: {
          grid: {
            display: false,
          },
          ticks: {
            font: {
              size: 11,
              family: "'Inter', sans-serif",
            },
            color: isDark ? "#9ca3af" : "#6b7280",
          },
        },
        y: {
          grid: {
            color: isDark ? "#374151" : "#e5e7eb",
          },
          ticks: {
            font: {
              size: 11,
              family: "'Inter', sans-serif",
            },
            color: isDark ? "#9ca3af" : "#6b7280",
            padding: 8,
          },
          border: {
            dash: [5, 5],
          },
        },
      };

      // Apply theme-specific styling to the datasets
      if (chartData?.datasets) {
        const updatedDatasets = chartData.datasets.map((dataset: any) => {
          return {
            ...dataset,
            backgroundColor: isDark ? "rgba(79, 70, 229, 0.7)" : "rgba(99, 102, 241, 0.7)",
            hoverBackgroundColor: isDark ? "rgb(79, 70, 229)" : "rgb(99, 102, 241)",
            borderRadius: 4,
            borderSkipped: false,
            barPercentage: 0.6,
            categoryPercentage: 0.7,
          };
        });

        setChartData({
          ...chartData,
          datasets: updatedDatasets,
        });
      }
    }

    if (type === "doughnut") {
      typeSpecificOptions.cutout = "70%";
      typeSpecificOptions.animation = {
        animateRotate: true,
        animateScale: true,
      };

      // Apply theme-specific styling to the doughnut
      if (chartData?.datasets) {
        setChartData({
          ...chartData,
          datasets: chartData.datasets.map((dataset: any) => ({
            ...dataset,
            borderWidth: 2,
            borderColor: isDark ? "#1f2937" : "#ffffff",
            hoverBorderColor: isDark ? "#1f2937" : "#ffffff",
            hoverOffset: 4,
          })),
        });
      }
    }

    setChartOptions({
      ...baseOptions,
      ...typeSpecificOptions,
    });
  }, [resolvedTheme, type, data, showLegend, chartData?.datasets]);

  // Render the appropriate chart type
  const renderChart = () => {
    switch (type) {
      case "line":
        return <Line data={chartData} options={chartOptions} height={height} />;
      case "bar":
        return <Bar data={chartData} options={chartOptions} height={height} />;
      case "doughnut":
        return <Doughnut data={chartData} options={chartOptions} height={height} />;
      default:
        return <Line data={chartData} options={chartOptions} height={height} />;
    }
  };

  return (
    <div
      className={cn(
        "bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 p-5 rounded-lg shadow-sm",
        className
      )}
    >
      <h3 className="text-base font-medium text-gray-900 dark:text-white mb-4">{title}</h3>
      <div style={{ height: `${height}px` }} className="relative">
        {renderChart()}
      </div>
    </div>
  );
}; 