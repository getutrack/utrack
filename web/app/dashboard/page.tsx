"use client";

import React from "react";
import { DashboardChart } from "@/components/charts/dashboard-chart";
import { CollapsibleSidebar } from "@/components/sidebar/collapsible-sidebar";
import { Breadcrumbs } from "@/components/breadcrumbs/breadcrumbs";
import { ButtonHoverEffect } from "@/components/interactions/button-hover-effect";
import { Users, TrendingUp, CreditCard, Activity } from "lucide-react";

const DashboardPage = () => {
  // Sample data for line chart
  const lineChartData = {
    labels: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"],
    datasets: [
      {
        label: "Active Users",
        data: [65, 59, 80, 81, 56, 55, 73],
      },
      {
        label: "New Users",
        data: [28, 48, 40, 19, 36, 27, 40],
        borderColor: "rgb(75, 192, 192)",
        tension: 0.1,
      },
    ],
  };

  // Sample data for bar chart
  const barChartData = {
    labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    datasets: [
      {
        label: "Revenue",
        data: [1200, 1900, 3000, 5000, 2000, 3000, 7000],
      },
    ],
  };

  // Sample data for doughnut chart
  const doughnutChartData = {
    labels: ["Completed", "In Progress", "Planned", "Cancelled"],
    datasets: [
      {
        label: "Project Status",
        data: [300, 150, 100, 50],
        backgroundColor: [
          "rgba(75, 192, 192, 0.8)",
          "rgba(54, 162, 235, 0.8)",
          "rgba(255, 206, 86, 0.8)",
          "rgba(255, 99, 132, 0.8)",
        ],
      },
    ],
  };

  // Stats for summary cards
  const stats = [
    { name: "Total Users", value: "12,345", icon: Users, change: "+12.3%" },
    { name: "Growth", value: "23.5%", icon: TrendingUp, change: "+5.7%" },
    { name: "Revenue", value: "$45,678", icon: CreditCard, change: "+10.2%" },
    { name: "Activity", value: "2,345", icon: Activity, change: "+3.1%" },
  ];

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
      {/* Sidebar */}
      <CollapsibleSidebar />

      {/* Main content */}
      <main className="flex-1 overflow-auto p-4 md:p-6">
        <div className="max-w-7xl mx-auto">
          {/* Breadcrumbs and header */}
          <div className="mb-6">
            <Breadcrumbs />
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
            <p className="text-gray-500 dark:text-gray-400">An overview of your application stats and activity</p>
          </div>

          {/* Stat Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
            {stats.map((stat) => (
              <ButtonHoverEffect key={stat.name}>
                <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                  <div className="flex justify-between items-center">
                    <div>
                      <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{stat.name}</p>
                      <p className="text-2xl font-bold text-gray-900 dark:text-white">{stat.value}</p>
                    </div>
                    <div className="w-12 h-12 rounded-lg bg-primary-50 dark:bg-primary-900/20 flex items-center justify-center text-primary-600 dark:text-primary-400">
                      <stat.icon className="w-6 h-6" />
                    </div>
                  </div>
                  <div className="mt-4">
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400">
                      {stat.change}
                    </span>
                    <span className="ml-2 text-xs text-gray-500 dark:text-gray-400">vs last period</span>
                  </div>
                </div>
              </ButtonHoverEffect>
            ))}
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <DashboardChart
              title="User Growth"
              type="line"
              data={lineChartData}
              height={300}
            />
            <DashboardChart
              title="Weekly Revenue"
              type="bar"
              data={barChartData}
              height={300}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <DashboardChart
                title="Monthly Performance"
                type="line"
                data={lineChartData}
                height={350}
              />
            </div>
            <div>
              <DashboardChart
                title="Project Status"
                type="doughnut"
                data={doughnutChartData}
                height={350}
              />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default DashboardPage; 