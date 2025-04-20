"use client";

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../helpers";
import {
  ECardDirection,
  ECardSpacing,
  ECardVariant,
  getCardStyle,
  TCardDirection,
  TCardSpacing,
  TCardVariant,
} from "./helper";

const cardVariants = cva(
  "rounded-lg border shadow-sm transition-all duration-200",
  {
    variants: {
      variant: {
        default: "bg-white border-gray-200 dark:border-gray-800 dark:bg-gray-950",
        outline: "border border-gray-200 bg-transparent dark:border-gray-800",
        primary: "border-primary-100 bg-primary-50 dark:border-primary-900 dark:bg-primary-950",
        success: "border-success-100 bg-success-50 dark:border-success-900 dark:bg-success-950",
        warning: "border-warning-100 bg-warning-50 dark:border-warning-900 dark:bg-warning-950",
        error: "border-error-100 bg-error-50 dark:border-error-900 dark:bg-error-950",
      },
      hover: {
        none: "",
        default: "hover:border-gray-300 dark:hover:border-gray-700 hover:shadow-md",
        lift: "hover:shadow-md hover:-translate-y-1",
        glow: "hover:shadow-[0_0_20px_rgba(99,102,241,0.2)]",
      },
    },
    defaultVariants: {
      variant: "default",
      hover: "default",
    },
  }
);

export type CardPropsVariants = VariantProps<typeof cardVariants>;

export interface CardProps extends Omit<React.HTMLAttributes<HTMLDivElement>, 'variant'>, CardPropsVariants {
  spacing?: TCardSpacing;
  direction?: TCardDirection;
  className?: string;
  children: React.ReactNode;
  onClick?: () => void;
  hoverEffect?: boolean;
}

const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant, hover, ...props }, ref) => (
    <div ref={ref} className={cn(cardVariants({ variant, hover, className }))} {...props} />
  )
);
Card.displayName = "Card";

const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-col space-y-1.5 p-6", className)}
    {...props}
  />
));
CardHeader.displayName = "CardHeader";

const CardTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn("text-lg font-semibold leading-none tracking-tight", className)}
    {...props}
  />
));
CardTitle.displayName = "CardTitle";

const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn("text-sm text-gray-500 dark:text-gray-400", className)}
    {...props}
  />
));
CardDescription.displayName = "CardDescription";

const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
));
CardContent.displayName = "CardContent";

const CardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex items-center p-6 pt-0", className)}
    {...props}
  />
));
CardFooter.displayName = "CardFooter";

export { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent };
