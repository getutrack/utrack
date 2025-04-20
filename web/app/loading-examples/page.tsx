"use client";

import React, { useState, useEffect } from "react";
import {
  Skeleton,
  SkeletonText,
  SkeletonCard,
  SkeletonAvatar,
  SkeletonTable,
} from "@/components/skeleton/skeleton";
import { Breadcrumbs } from "@/components/breadcrumbs/breadcrumbs";
import { CollapsibleSidebar } from "@/components/sidebar/collapsible-sidebar";
import { ButtonHoverEffect } from "@/components/interactions/button-hover-effect";

const LoadingExamplesPage = () => {
  const [loading, setLoading] = useState(true);
  const [contentType, setContentType] = useState<"cards" | "table" | "profile">("cards");

  // Simulate a loading state
  useEffect(() => {
    const timer = setTimeout(() => {
      setLoading(false);
    }, 3000);

    return () => clearTimeout(timer);
  }, [contentType]);

  // Reset loading when content type changes
  useEffect(() => {
    setLoading(true);
  }, [contentType]);

  const handleContentTypeChange = (type: typeof contentType) => {
    setContentType(type);
  };

  // Example content for different content types
  const renderContent = () => {
    if (loading) {
      switch (contentType) {
        case "cards":
          return (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Array.from({ length: 6 }).map((_, i) => (
                <SkeletonCard key={i} />
              ))}
            </div>
          );
        case "table":
          return <SkeletonTable rows={8} cols={4} className="w-full" />;
        case "profile":
          return (
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center space-x-4 mb-6">
                <SkeletonAvatar size={64} />
                <div className="flex-1">
                  <Skeleton width="60%" height={24} className="mb-2" />
                  <Skeleton width="40%" height={16} />
                </div>
              </div>
              <SkeletonText lines={4} className="mb-6" />
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Skeleton height={40} />
                <Skeleton height={40} />
              </div>
            </div>
          );
        default:
          return null;
      }
    }

    // Show actual content when loading is done
    switch (contentType) {
      case "cards":
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {Array.from({ length: 6 }).map((_, i) => (
              <div
                key={i}
                className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700"
              >
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">Card Title {i + 1}</h3>
                <p className="text-gray-500 dark:text-gray-400 mb-4">
                  This is a sample card with demo content. Cards can be used to display various types of information.
                </p>
                <div className="flex justify-between items-center">
                  <button className="text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300 font-medium">
                    View Details
                  </button>
                  <div className="text-sm text-gray-500 dark:text-gray-400">3 days ago</div>
                </div>
              </div>
            ))}
          </div>
        );
      case "table":
        return (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-900">
                <tr>
                  {["Name", "Status", "Date", "Actions"].map((header) => (
                    <th
                      key={header}
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
                    >
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {Array.from({ length: 8 }).map((_, i) => (
                  <tr key={i}>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="flex-shrink-0 h-10 w-10 rounded-full bg-primary-100 dark:bg-primary-900/20 flex items-center justify-center text-primary-600 dark:text-primary-400">
                          {String.fromCharCode(65 + i)}
                        </div>
                        <div className="ml-4">
                          <div className="text-sm font-medium text-gray-900 dark:text-white">User {i + 1}</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">user{i + 1}@example.com</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          i % 3 === 0
                            ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400"
                            : i % 3 === 1
                            ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400"
                            : "bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400"
                        }`}
                      >
                        {i % 3 === 0 ? "Active" : i % 3 === 1 ? "Pending" : "Inactive"}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {new Date(Date.now() - i * 86400000).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <button className="text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300 mr-3">
                        Edit
                      </button>
                      <button className="text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300">
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      case "profile":
        return (
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center space-x-4 mb-6">
              <div className="w-16 h-16 rounded-full bg-primary-100 dark:bg-primary-900/20 flex items-center justify-center text-primary-600 dark:text-primary-400 text-xl font-medium">
                JD
              </div>
              <div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">John Doe</h3>
                <p className="text-gray-500 dark:text-gray-400">Senior Developer</p>
              </div>
            </div>

            <div className="mb-6">
              <p className="text-gray-700 dark:text-gray-300 mb-4">
                Frontend developer with experience in React, Vue, and Angular. Passionate about creating clean, efficient
                UIs and improving user experience.
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                Currently working on modernizing legacy applications and implementing design systems.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <button className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors">
                Message
              </button>
              <button className="px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors">
                View Projects
              </button>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
      {/* Sidebar */}
      <CollapsibleSidebar />

      {/* Main content */}
      <main className="flex-1 overflow-auto p-4 md:p-6">
        <div className="max-w-7xl mx-auto">
          {/* Breadcrumbs and header */}
          <div className="mb-6">
            <Breadcrumbs titleMap={{ "loading-examples": "Loading Examples" }} />
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Skeleton Loading Examples</h1>
            <p className="text-gray-500 dark:text-gray-400">Examples of skeleton loading states for different content types</p>
          </div>

          {/* Controls */}
          <div className="mb-6 flex flex-wrap gap-3">
            <ButtonHoverEffect>
              <button
                className={`px-4 py-2 rounded-md ${
                  contentType === "cards"
                    ? "bg-primary-600 text-white"
                    : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700"
                }`}
                onClick={() => handleContentTypeChange("cards")}
              >
                Cards
              </button>
            </ButtonHoverEffect>
            <ButtonHoverEffect>
              <button
                className={`px-4 py-2 rounded-md ${
                  contentType === "table"
                    ? "bg-primary-600 text-white"
                    : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700"
                }`}
                onClick={() => handleContentTypeChange("table")}
              >
                Table
              </button>
            </ButtonHoverEffect>
            <ButtonHoverEffect>
              <button
                className={`px-4 py-2 rounded-md ${
                  contentType === "profile"
                    ? "bg-primary-600 text-white"
                    : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700"
                }`}
                onClick={() => handleContentTypeChange("profile")}
              >
                Profile
              </button>
            </ButtonHoverEffect>
            <ButtonHoverEffect>
              <button
                className="px-4 py-2 rounded-md bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700"
                onClick={() => setLoading(true)}
              >
                Reload
              </button>
            </ButtonHoverEffect>
          </div>

          {/* Content area */}
          <div className="my-6">{renderContent()}</div>
        </div>
      </main>
    </div>
  );
};

export default LoadingExamplesPage; 