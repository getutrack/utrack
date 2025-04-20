"use client";

import React, { useState } from "react";
import { OnboardingTour, OnboardingStep } from "@/components/onboarding/onboarding-tour";
import { Breadcrumbs } from "@/components/breadcrumbs/breadcrumbs";
import { CollapsibleSidebar } from "@/components/sidebar/collapsible-sidebar";
import { ButtonHoverEffect } from "@/components/interactions/button-hover-effect";
import { Settings, Users, Bell, Search, Zap, HelpCircle } from "lucide-react";

const OnboardingPage = () => {
  const [isTourOpen, setIsTourOpen] = useState(false);
  const [currentTour, setCurrentTour] = useState<"app" | "features" | "advanced">("app");

  // Define tour steps for different tours
  const appTourSteps: OnboardingStep[] = [
    {
      id: "welcome",
      title: "Welcome to Utrack",
      description: "This quick tour will help you get familiar with the application. Let's get started!",
      placement: "bottom",
    },
    {
      id: "navigation",
      title: "Navigation",
      description: "The sidebar helps you navigate between different sections of the application.",
      target: "#sidebar-navigation",
      placement: "right",
      spotlightRadius: 150,
    },
    {
      id: "profile",
      title: "Your Profile",
      description: "Access your profile settings and account preferences here.",
      target: "#profile-section",
      placement: "top",
      spotlightRadius: 80,
    },
    {
      id: "search",
      title: "Global Search",
      description: "Quickly find what you're looking for with our powerful search.",
      target: "#search-bar",
      placement: "bottom",
      spotlightRadius: 100,
    },
    {
      id: "complete",
      title: "You're All Set!",
      description: "You've completed the basic tour. Feel free to explore the application on your own.",
      placement: "bottom",
    },
  ];

  const featuresTourSteps: OnboardingStep[] = [
    {
      id: "features-welcome",
      title: "Feature Tour",
      description: "Let's explore some of the core features that will help you be more productive.",
      placement: "bottom",
    },
    {
      id: "dashboard",
      title: "Dashboard",
      description: "Get an overview of your projects, tasks, and team activity at a glance.",
      target: "#dashboard-section",
      placement: "bottom",
      spotlightRadius: 120,
    },
    {
      id: "analytics",
      title: "Analytics",
      description: "Dive deep into your data with interactive charts and reports.",
      target: "#analytics-section",
      placement: "bottom",
      spotlightRadius: 120,
    },
    {
      id: "notifications",
      title: "Notifications",
      description: "Stay up-to-date with important updates and alerts.",
      target: "#notifications-section",
      placement: "left",
      spotlightRadius: 80,
    },
    {
      id: "features-complete",
      title: "Features Tour Completed",
      description: "You now know about the main features. Ready to explore more?",
      placement: "bottom",
    },
  ];

  const advancedTourSteps: OnboardingStep[] = [
    {
      id: "advanced-welcome",
      title: "Advanced Features",
      description: "Let's discover some advanced features to help you get the most out of Utrack.",
      placement: "bottom",
    },
    {
      id: "settings",
      title: "Advanced Settings",
      description: "Customize the application to match your workflow preferences.",
      target: "#settings-section",
      placement: "left",
      spotlightRadius: 80,
    },
    {
      id: "team",
      title: "Team Management",
      description: "Manage your team members and their permissions efficiently.",
      target: "#team-section",
      placement: "bottom",
      spotlightRadius: 120,
    },
    {
      id: "shortcuts",
      title: "Keyboard Shortcuts",
      description: "Speed up your workflow with convenient keyboard shortcuts.",
      target: "#shortcuts-section",
      placement: "right",
      spotlightRadius: 100,
    },
    {
      id: "advanced-complete",
      title: "You're a Pro!",
      description: "You've mastered the advanced features of Utrack. Enjoy using the application!",
      placement: "bottom",
    },
  ];

  // Get the current tour steps based on selection
  const getCurrentTourSteps = () => {
    switch (currentTour) {
      case "app":
        return appTourSteps;
      case "features":
        return featuresTourSteps;
      case "advanced":
        return advancedTourSteps;
      default:
        return appTourSteps;
    }
  };

  const startTour = (tour: typeof currentTour) => {
    setCurrentTour(tour);
    setIsTourOpen(true);
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
            <Breadcrumbs titleMap={{ onboarding: "Onboarding" }} />
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Onboarding Tour</h1>
            <p className="text-gray-500 dark:text-gray-400">
              Examples of guided onboarding tours to help users learn about the application
            </p>
          </div>

          {/* Tour selection */}
          <div className="mb-10 bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Start a Tour</h2>
            <p className="text-gray-600 dark:text-gray-400 mb-6">
              Choose one of the available tours to see how guided onboarding helps users discover your application's
              features.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <ButtonHoverEffect>
                <button
                  onClick={() => startTour("app")}
                  className="w-full p-4 flex flex-col items-center bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors"
                >
                  <Zap className="h-8 w-8 text-primary-600 dark:text-primary-400 mb-3" />
                  <h3 className="text-gray-900 dark:text-white font-medium">App Tour</h3>
                  <p className="text-gray-500 dark:text-gray-400 text-sm text-center mt-2">
                    General introduction to the application interface
                  </p>
                </button>
              </ButtonHoverEffect>

              <ButtonHoverEffect>
                <button
                  onClick={() => startTour("features")}
                  className="w-full p-4 flex flex-col items-center bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors"
                >
                  <HelpCircle className="h-8 w-8 text-primary-600 dark:text-primary-400 mb-3" />
                  <h3 className="text-gray-900 dark:text-white font-medium">Features Tour</h3>
                  <p className="text-gray-500 dark:text-gray-400 text-sm text-center mt-2">
                    Learn about the core features and functionality
                  </p>
                </button>
              </ButtonHoverEffect>

              <ButtonHoverEffect>
                <button
                  onClick={() => startTour("advanced")}
                  className="w-full p-4 flex flex-col items-center bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors"
                >
                  <Settings className="h-8 w-8 text-primary-600 dark:text-primary-400 mb-3" />
                  <h3 className="text-gray-900 dark:text-white font-medium">Advanced Tour</h3>
                  <p className="text-gray-500 dark:text-gray-400 text-sm text-center mt-2">
                    Discover advanced features and customization options
                  </p>
                </button>
              </ButtonHoverEffect>
            </div>
          </div>

          {/* Demo UI elements for tour targets */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Left column */}
            <div>
              <div
                id="sidebar-navigation"
                className="mb-6 bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700"
              >
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-3">Navigation</h3>
                <ul className="space-y-2">
                  <li>
                    <a
                      href="#"
                      id="dashboard-section"
                      className="flex items-center p-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md"
                    >
                      <Zap className="h-5 w-5 mr-3 text-gray-500 dark:text-gray-400" />
                      Dashboard
                    </a>
                  </li>
                  <li>
                    <a
                      href="#"
                      id="analytics-section"
                      className="flex items-center p-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md"
                    >
                      <Bell className="h-5 w-5 mr-3 text-gray-500 dark:text-gray-400" />
                      Analytics
                    </a>
                  </li>
                  <li>
                    <a
                      href="#"
                      id="team-section"
                      className="flex items-center p-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md"
                    >
                      <Users className="h-5 w-5 mr-3 text-gray-500 dark:text-gray-400" />
                      Team
                    </a>
                  </li>
                  <li>
                    <a
                      href="#"
                      id="settings-section"
                      className="flex items-center p-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md"
                    >
                      <Settings className="h-5 w-5 mr-3 text-gray-500 dark:text-gray-400" />
                      Settings
                    </a>
                  </li>
                </ul>
              </div>
            </div>

            {/* Middle column */}
            <div className="md:col-span-2">
              <div className="mb-6 bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white">Dashboard Overview</h3>
                  <div className="flex space-x-2">
                    <div id="profile-section" className="relative">
                      <button className="w-8 h-8 bg-primary-100 dark:bg-primary-900/20 rounded-full flex items-center justify-center text-primary-600 dark:text-primary-400">
                        U
                      </button>
                    </div>
                    <div id="notifications-section" className="relative">
                      <button className="w-8 h-8 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center text-gray-600 dark:text-gray-300">
                        <Bell className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                </div>

                <div id="search-bar" className="mb-6 relative">
                  <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                    <Search className="h-4 w-4 text-gray-400" />
                  </div>
                  <input
                    type="search"
                    className="block w-full pl-10 pr-3 py-2 rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-primary-500 focus:border-primary-500"
                    placeholder="Search..."
                  />
                </div>

                <div id="shortcuts-section" className="mb-4 p-3 bg-gray-50 dark:bg-gray-700 rounded-md">
                  <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Keyboard Shortcuts</h4>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Search</span>
                      <span className="font-mono bg-gray-200 dark:bg-gray-600 px-1 rounded">Ctrl+K</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">New Item</span>
                      <span className="font-mono bg-gray-200 dark:bg-gray-600 px-1 rounded">Ctrl+N</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Save</span>
                      <span className="font-mono bg-gray-200 dark:bg-gray-600 px-1 rounded">Ctrl+S</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Help</span>
                      <span className="font-mono bg-gray-200 dark:bg-gray-600 px-1 rounded">F1</span>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-md">
                    <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Recent Activity</h4>
                    <div className="space-y-2 text-xs text-gray-600 dark:text-gray-400">
                      <div>Updated Project A - 2h ago</div>
                      <div>Added new task - 3h ago</div>
                      <div>Completed milestone - 1d ago</div>
                    </div>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-md">
                    <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Team Members</h4>
                    <div className="flex -space-x-2">
                      {["A", "B", "C", "D"].map((letter, i) => (
                        <div
                          key={i}
                          className="w-6 h-6 rounded-full flex items-center justify-center text-xs text-white"
                          style={{
                            backgroundColor: [`#4f46e5`, `#10b981`, `#f59e0b`, `#ef4444`][i],
                            zIndex: 4 - i,
                          }}
                        >
                          {letter}
                        </div>
                      ))}
                      <div className="w-6 h-6 rounded-full bg-gray-300 dark:bg-gray-600 flex items-center justify-center text-xs text-gray-800 dark:text-gray-200 z-0">
                        +3
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Onboarding Tour */}
          <OnboardingTour
            steps={getCurrentTourSteps()}
            isOpen={isTourOpen}
            onClose={() => setIsTourOpen(false)}
            onComplete={() => setIsTourOpen(false)}
            highlightTarget={true}
          />
        </div>
      </main>
    </div>
  );
};

export default OnboardingPage;
