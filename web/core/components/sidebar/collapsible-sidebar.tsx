"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronLeft, ChevronRight, Home, Layout, Calendar, Settings, Users, BarChart2 } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/helpers/common.helper";

const sidebarVariants = {
  expanded: {
    width: "240px",
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 30
    }
  },
  collapsed: {
    width: "64px",
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 30
    }
  }
};

const itemVariants = {
  expanded: {
    opacity: 1,
    x: 0,
    transition: {
      duration: 0.2
    }
  },
  collapsed: {
    opacity: 0,
    x: -10,
    transition: {
      duration: 0.2
    }
  }
};

const navItems = [
  { name: "Dashboard", icon: Home, href: "/dashboard" },
  { name: "Projects", icon: Layout, href: "/projects" },
  { name: "Calendar", icon: Calendar, href: "/calendar" },
  { name: "Analytics", icon: BarChart2, href: "/analytics" },
  { name: "Team", icon: Users, href: "/team" },
  { name: "Settings", icon: Settings, href: "/settings" },
];

interface CollapsibleSidebarProps {
  className?: string;
}

export const CollapsibleSidebar = ({ className }: CollapsibleSidebarProps) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const pathname = usePathname();

  return (
    <motion.aside
      variants={sidebarVariants}
      initial="expanded"
      animate={isExpanded ? "expanded" : "collapsed"}
      className={cn(
        "h-screen bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 flex flex-col z-30 shadow-sm",
        className
      )}
    >
      {/* Logo & Toggle */}
      <div className="flex items-center px-4 h-16 border-b border-gray-200 dark:border-gray-800">
        <AnimatePresence mode="wait">
          {isExpanded && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-lg font-semibold text-gray-900 dark:text-white"
            >
              Utrack
            </motion.div>
          )}
        </AnimatePresence>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className={cn(
            "ml-auto bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400 rounded-md p-1 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors",
            isExpanded ? "" : "mx-auto"
          )}
          aria-expanded={isExpanded}
          aria-label={isExpanded ? "Collapse sidebar" : "Expand sidebar"}
        >
          {isExpanded ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
        </button>
      </div>

      {/* Nav Items */}
      <nav className="flex-grow pt-5">
        <ul className="space-y-1 px-2">
          {navItems.map((item) => {
            const isActive = pathname.startsWith(item.href);
            
            return (
              <li key={item.name}>
                <Link 
                  href={item.href}
                  className={cn(
                    "flex items-center rounded-md px-3 py-2 transition-colors",
                    isActive 
                      ? "bg-primary-50 text-primary-700 dark:bg-primary-900/20 dark:text-primary-400" 
                      : "text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800"
                  )}
                  aria-current={isActive ? "page" : undefined}
                >
                  <item.icon className={cn("h-5 w-5", isActive ? "text-primary-600 dark:text-primary-400" : "text-gray-500 dark:text-gray-400")} />
                  
                  {isExpanded && (
                    <motion.span variants={itemVariants} className="ml-3 font-medium">{item.name}</motion.span>
                  )}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* User Profile */}
      <div className="p-3 border-t border-gray-200 dark:border-gray-800 flex items-center">
        <div className="w-8 h-8 rounded-full bg-primary-100 dark:bg-primary-800 flex items-center justify-center text-primary-600 dark:text-primary-300 font-medium">
          U
        </div>
        
        {isExpanded && (
          <motion.div variants={itemVariants} className="ml-3">
            <div className="text-sm font-medium text-gray-900 dark:text-white">User Name</div>
            <div className="text-xs text-gray-500 dark:text-gray-400">user@example.com</div>
          </motion.div>
        )}
      </div>
    </motion.aside>
  );
}; 