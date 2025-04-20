"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChevronRight, Home } from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/helpers/common.helper";

interface BreadcrumbsProps {
  homeHref?: string;
  className?: string;
  titleMap?: Record<string, string>;
  omitHome?: boolean;
}

export const Breadcrumbs = ({
  homeHref = "/dashboard",
  className,
  titleMap = {},
  omitHome = false,
}: BreadcrumbsProps) => {
  const pathname = usePathname();
  
  // Skip rendering if we're on the home page
  if (pathname === homeHref) return null;
  
  // Split the path into segments
  const pathSegments = pathname.split("/").filter(Boolean);
  
  // Construct the breadcrumb items with paths
  const breadcrumbs = pathSegments.map((segment, index) => {
    const path = `/${pathSegments.slice(0, index + 1).join("/")}`;
    // Replace hyphens with spaces and capitalize each word for title
    const defaultTitle = segment
      .split("-")
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
      
    // Use custom title from map if available
    const title = titleMap[segment] || defaultTitle;
    
    return { path, title, segment };
  });
  
  // Prepend home if not omitted
  if (!omitHome) {
    breadcrumbs.unshift({ path: homeHref, title: "Home", segment: "home" });
  }

  // Animation variants
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const item = {
    hidden: { opacity: 0, y: -10 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <motion.nav
      variants={container}
      initial="hidden"
      animate="show"
      aria-label="Breadcrumb"
      className={cn("py-3 flex items-center text-sm", className)}
    >
      <ol className="flex items-center space-x-1">
        {breadcrumbs.map((breadcrumb, index) => {
          const isLast = index === breadcrumbs.length - 1;
          
          return (
            <React.Fragment key={breadcrumb.path}>
              <motion.li variants={item}>
                {isLast ? (
                  <span
                    className="text-gray-700 dark:text-gray-300 font-medium"
                    aria-current="page"
                  >
                    {breadcrumb.title}
                  </span>
                ) : (
                  <Link
                    href={breadcrumb.path}
                    className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:underline transition-colors"
                  >
                    {index === 0 && !omitHome ? <Home className="h-3.5 w-3.5" /> : breadcrumb.title}
                  </Link>
                )}
              </motion.li>
              
              {!isLast && (
                <motion.li variants={item}>
                  <ChevronRight
                    className="h-3.5 w-3.5 text-gray-400 flex-shrink-0"
                    aria-hidden="true"
                  />
                </motion.li>
              )}
            </React.Fragment>
          );
        })}
      </ol>
    </motion.nav>
  );
}; 