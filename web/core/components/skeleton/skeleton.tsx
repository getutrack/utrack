"use client";

import React from "react";
import { VariantProps, cva } from "class-variance-authority";
import { cn } from "@/helpers/common.helper";

const skeletonVariants = cva(
  "animate-pulse bg-gray-200 dark:bg-gray-700 rounded-md",
  {
    variants: {
      variant: {
        default: "bg-gray-200 dark:bg-gray-700",
        primary: "bg-primary-100 dark:bg-primary-900",
      },
      size: {
        xs: "h-2",
        sm: "h-3",
        md: "h-4",
        lg: "h-6",
        xl: "h-8",
        "2xl": "h-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "md",
    },
  }
);

export interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement>, VariantProps<typeof skeletonVariants> {
  width?: string | number;
  height?: string | number;
  rounded?: "sm" | "md" | "lg" | "full";
}

export const Skeleton = ({
  className,
  variant,
  size,
  width,
  height,
  rounded = "md",
  ...props
}: SkeletonProps) => {
  // Style calculation
  const heightStyles = height ? { height: typeof height === 'number' ? `${height}px` : height } : {};
  const widthStyles = width ? { width: typeof width === 'number' ? `${width}px` : width } : {};

  // Custom rounded classes
  const roundedClasses = {
    sm: "rounded-sm",
    md: "rounded-md",
    lg: "rounded-lg",
    full: "rounded-full",
  };

  return (
    <div
      className={cn(
        skeletonVariants({ variant, size, className }),
        roundedClasses[rounded]
      )}
      style={{ ...heightStyles, ...widthStyles }}
      {...props}
    />
  );
};

// Predefined skeleton components
export const SkeletonText = ({ className, lines = 1, ...props }: { className?: string; lines?: number } & Omit<SkeletonProps, "width">) => {
  return (
    <div className={cn("space-y-2", className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton 
          key={i} 
          width={i === lines - 1 && lines > 1 ? "75%" : "100%"} 
          size="sm" 
          {...props} 
        />
      ))}
    </div>
  );
};

export const SkeletonCard = ({ className, ...props }: SkeletonProps) => (
  <div className={cn("p-4 border border-gray-200 dark:border-gray-800 rounded-lg", className)}>
    <Skeleton width="40%" height={24} className="mb-4" {...props} />
    <SkeletonText lines={3} className="mb-4" {...props} />
    <div className="flex items-center justify-between">
      <Skeleton width={100} height={32} rounded="md" {...props} />
      <Skeleton width={24} height={24} rounded="full" {...props} />
    </div>
  </div>
);

export const SkeletonAvatar = ({ size = 40, className, ...props }: { size?: number } & Omit<SkeletonProps, "size">) => (
  <Skeleton 
    width={size} 
    height={size} 
    rounded="full" 
    className={className}
    {...props} 
  />
);

export const SkeletonTable = ({ rows = 5, cols = 4, className, ...props }: { rows?: number; cols?: number } & SkeletonProps) => (
  <div className={cn("space-y-3", className)}>
    <div className="flex items-center space-x-3">
      {Array.from({ length: cols }).map((_, i) => (
        <Skeleton key={i} width={`${Math.floor(90 / cols)}%`} height={28} {...props} />
      ))}
    </div>
    {Array.from({ length: rows }).map((_, i) => (
      <div key={i} className="flex items-center space-x-3">
        {Array.from({ length: cols }).map((_, j) => (
          <Skeleton key={j} width={`${Math.floor(90 / cols)}%`} height={16} {...props} />
        ))}
      </div>
    ))}
  </div>
); 