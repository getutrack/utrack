"use client";

import React, { ReactNode } from "react";
import Image from "next/image";
import { Button } from "@utrack/ui";
import { motion } from "framer-motion";
import { cn } from "@/helpers/common.helper";

// Import illustrations
import EmptyInbox from "@/public/illustrations/empty-inbox.svg";
import EmptySearch from "@/public/illustrations/empty-search.svg";
import NoResults from "@/public/illustrations/no-results.svg";
import NoData from "@/public/illustrations/no-data.svg";
import NoTasks from "@/public/illustrations/no-tasks.svg";
import CompleteTask from "@/public/illustrations/complete-task.svg";

export const illustrations = {
  "empty-inbox": EmptyInbox,
  "empty-search": EmptySearch,
  "no-results": NoResults,
  "no-data": NoData,
  "no-tasks": NoTasks,
  "complete-task": CompleteTask,
};

export type IllustrationType = keyof typeof illustrations;

interface EmptyStateProps {
  title: string;
  description?: ReactNode;
  illustration?: IllustrationType;
  primaryAction?: {
    label: string;
    onClick: () => void;
    variant?: "primary" | "secondary" | "outline";
  };
  secondaryAction?: {
    label: string;
    onClick: () => void;
    variant?: "outline" | "ghost" | "link";
  };
  className?: string;
  isCompact?: boolean;
  illustrationSize?: "sm" | "md" | "lg";
}

export const EmptyState = ({
  title,
  description,
  illustration = "no-data",
  primaryAction,
  secondaryAction,
  className,
  isCompact = false,
  illustrationSize = "md",
}: EmptyStateProps) => {
  const IllustrationComponent = illustrations[illustration];

  // Illustration sizes
  const illustrationSizes = {
    sm: { width: 120, height: 120, className: "my-4" },
    md: { width: 180, height: 180, className: "my-6" },
    lg: { width: 240, height: 240, className: "my-8" },
  };

  const { width, height, className: sizeClassName } = illustrationSizes[illustrationSize];

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        when: "beforeChildren",
        staggerChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className={cn(
        "flex flex-col items-center justify-center text-center",
        isCompact ? "p-6" : "p-10 md:p-16",
        className
      )}
    >
      <motion.div variants={itemVariants} className={sizeClassName}>
        <Image
          src={IllustrationComponent}
          alt={title}
          width={width}
          height={height}
          priority
          className="transition-transform duration-500 hover:scale-105"
        />
      </motion.div>

      <motion.h3
        variants={itemVariants}
        className="text-xl font-semibold text-gray-900 dark:text-white mb-2"
      >
        {title}
      </motion.h3>

      {description && (
        <motion.div
          variants={itemVariants}
          className="text-gray-500 dark:text-gray-400 mb-6 max-w-md"
        >
          {description}
        </motion.div>
      )}

      {(primaryAction || secondaryAction) && (
        <motion.div variants={itemVariants} className="flex flex-wrap gap-3 justify-center">
          {primaryAction && (
            <Button
              variant={primaryAction.variant || "primary"}
              onClick={primaryAction.onClick}
            >
              {primaryAction.label}
            </Button>
          )}
          {secondaryAction && (
            <Button
              variant={secondaryAction.variant || "outline"}
              onClick={secondaryAction.onClick}
            >
              {secondaryAction.label}
            </Button>
          )}
        </motion.div>
      )}
    </motion.div>
  );
}; 