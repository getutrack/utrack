"use client";

import React, { useState } from "react";
import { EmptyState, IllustrationType } from "@/components/empty-states/empty-state";
import { Breadcrumbs } from "@/components/breadcrumbs/breadcrumbs";
import { CollapsibleSidebar } from "@/components/sidebar/collapsible-sidebar";
import { ButtonHoverEffect } from "@/components/interactions/button-hover-effect";

const EmptyStatesPage = () => {
  const [selectedState, setSelectedState] = useState<IllustrationType>("no-data");
  const [isCompact, setIsCompact] = useState(false);

  const emptyStateOptions: Array<{
    type: IllustrationType;
    title: string;
    description: string;
  }> = [
    {
      type: "no-data",
      title: "No data available",
      description: "There's no data available for this view. Add some data to get started.",
    },
    {
      type: "empty-inbox",
      title: "Your inbox is empty",
      description: "You've handled all your messages. Take a break or check back later.",
    },
    {
      type: "empty-search",
      title: "No search results",
      description: "We couldn't find any results matching your search. Try another search term.",
    },
    {
      type: "no-results",
      title: "No results found",
      description: "There are no items matching your filters. Try changing your filter criteria.",
    },
    {
      type: "no-tasks",
      title: "No tasks yet",
      description: "You don't have any tasks assigned to you. Time to create your first task!",
    },
    {
      type: "complete-task",
      title: "All tasks completed",
      description: "Great job! You've completed all your assigned tasks. Time to celebrate!",
    },
  ];

  const currentState = emptyStateOptions.find((state) => state.type === selectedState) || emptyStateOptions[0];

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
      {/* Sidebar */}
      <CollapsibleSidebar />

      {/* Main content */}
      <main className="flex-1 overflow-auto p-4 md:p-6">
        <div className="max-w-7xl mx-auto">
          {/* Breadcrumbs and header */}
          <div className="mb-6">
            <Breadcrumbs titleMap={{ "empty-states": "Empty States" }} />
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Empty States</h1>
            <p className="text-gray-500 dark:text-gray-400">Examples of empty state components for better user experience</p>
          </div>

          {/* Controls */}
          <div className="mb-6 bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Customize Empty State</h2>
            <div className="space-y-4">
              <div>
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Select Illustration Type:</p>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
                  {emptyStateOptions.map((option) => (
                    <ButtonHoverEffect key={option.type}>
                      <button
                        className={`w-full px-3 py-2 text-xs rounded-md ${
                          selectedState === option.type
                            ? "bg-primary-600 text-white"
                            : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700"
                        }`}
                        onClick={() => setSelectedState(option.type)}
                      >
                        {option.type.split("-").join(" ")}
                      </button>
                    </ButtonHoverEffect>
                  ))}
                </div>
              </div>

              <div className="flex items-center">
                <input
                  id="compact-mode"
                  type="checkbox"
                  checked={isCompact}
                  onChange={() => setIsCompact(!isCompact)}
                  className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                />
                <label htmlFor="compact-mode" className="ml-2 block text-sm text-gray-900 dark:text-gray-100">
                  Compact Mode
                </label>
              </div>
            </div>
          </div>

          {/* Preview */}
          <div className="mb-6">
            <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Preview</h2>
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
              <EmptyState
                title={currentState.title}
                description={currentState.description}
                illustration={currentState.type}
                isCompact={isCompact}
                illustrationSize={isCompact ? "sm" : "lg"}
                primaryAction={{
                  label: "Primary Action",
                  onClick: () => alert("Primary action clicked"),
                  variant: "primary",
                }}
                secondaryAction={{
                  label: "Secondary Action",
                  onClick: () => alert("Secondary action clicked"),
                  variant: "outline",
                }}
              />
            </div>
          </div>

          {/* Information */}
          <div className="mt-10">
            <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-4">About Empty States</h2>
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-gray-700 dark:text-gray-300 mb-4">
                Empty states are an important part of user experience design. They help users understand what to expect
                when no content is available and provide guidance on how to proceed.
              </p>
              <p className="text-gray-700 dark:text-gray-300 mb-4">
                Good empty states should:
              </p>
              <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-2 mb-4">
                <li>Clearly explain why content is missing</li>
                <li>Use appropriate illustrations to convey the message</li>
                <li>Provide helpful actions for users to take</li>
                <li>Maintain the overall design aesthetic of your application</li>
                <li>Be friendly and encouraging</li>
              </ul>
              <p className="text-gray-700 dark:text-gray-300">
                The empty state components in this example follow these principles, providing a better user experience
                when content is not available.
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default EmptyStatesPage; 