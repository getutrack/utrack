"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, ChevronRight, ChevronLeft } from "lucide-react";
import { Button } from "@utrack/ui";
import { cn } from "@/helpers/common.helper";

export interface OnboardingStep {
  id: string;
  title: string;
  description: React.ReactNode;
  target?: string; // CSS selector for the target element
  placement?: "top" | "right" | "bottom" | "left";
  spotlightRadius?: number;
  offset?: number;
  actions?: {
    skipLabel?: string;
    nextLabel?: string;
    prevLabel?: string;
    doneLabel?: string;
  };
}

interface OnboardingTourProps {
  steps: OnboardingStep[];
  isOpen: boolean;
  onClose: () => void;
  onComplete: () => void;
  className?: string;
  disableOverlayClose?: boolean;
  highlightTarget?: boolean;
  initialStep?: number;
}

export const OnboardingTour = ({
  steps,
  isOpen,
  onClose,
  onComplete,
  className,
  disableOverlayClose = false,
  highlightTarget = true,
  initialStep = 0,
}: OnboardingTourProps) => {
  const [currentStep, setCurrentStep] = useState(initialStep);
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });
  const [windowSize, setWindowSize] = useState({ width: 0, height: 0 });

  // Get the current step data
  const step = steps[currentStep];

  // Handle resize and recalculate positions
  useEffect(() => {
    if (!isOpen) return;

    const handleResize = () => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    handleResize(); // Initial calculation
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [isOpen]);

  // Calculate position when step or window size changes
  useEffect(() => {
    if (!isOpen || !step.target) return;

    const calculatePosition = () => {
      const targetElement = document.querySelector(step.target!);
      if (!targetElement) return;

      const offset = step.offset || 20;
      const rect = targetElement.getBoundingClientRect();
      const placement = step.placement || "bottom";

      let top = 0;
      let left = 0;

      switch (placement) {
        case "top":
          top = rect.top - offset;
          left = rect.left + rect.width / 2;
          break;
        case "right":
          top = rect.top + rect.height / 2;
          left = rect.right + offset;
          break;
        case "bottom":
          top = rect.bottom + offset;
          left = rect.left + rect.width / 2;
          break;
        case "left":
          top = rect.top + rect.height / 2;
          left = rect.left - offset;
          break;
      }

      // Add any scrolling offset
      top += window.scrollY;
      left += window.scrollX;

      setTooltipPosition({ top, left });

      // If highlighting is enabled, scroll to make the element visible
      if (highlightTarget) {
        targetElement.scrollIntoView({
          behavior: "smooth",
          block: "center",
        });
      }
    };

    // Small delay to ensure DOM is ready
    const timer = setTimeout(calculatePosition, 100);
    return () => clearTimeout(timer);
  }, [currentStep, windowSize, isOpen, step, highlightTarget]);

  // Handle navigation
  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      handleComplete();
    }
  };

  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSkip = () => {
    onClose();
  };

  const handleComplete = () => {
    onComplete();
    onClose();
  };

  // Animation variants
  const overlayVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { duration: 0.3 } },
  };

  const tooltipVariants = {
    hidden: { opacity: 0, scale: 0.8 },
    visible: { 
      opacity: 1, 
      scale: 1, 
      transition: { 
        type: "spring", 
        stiffness: 300, 
        damping: 25 
      } 
    },
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial="hidden"
          animate="visible"
          exit="hidden"
          variants={overlayVariants}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 backdrop-blur-sm"
          onClick={disableOverlayClose ? undefined : handleSkip}
        >
          {/* Spotlight effect for the target element */}
          {step.target && highlightTarget && (
            <div
              className="absolute bg-black bg-opacity-50 inset-0"
              style={{
                mask: step.target
                  ? `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100%' height='100%'%3E%3Cdefs%3E%3Cmask id='mask' x='0' y='0' width='100%' height='100%'%3E%3Crect x='0' y='0' width='100%' height='100%' fill='white'/%3E%3Ccircle cx='${tooltipPosition.left}' cy='${tooltipPosition.top - window.scrollY}' r='${step.spotlightRadius || 100}' fill='black'/%3E%3C/mask%3E%3C/defs%3E%3Crect x='0' y='0' width='100%' height='100%' mask='url(%23mask)' fill='black'/%3E%3C/svg%3E")`
                  : undefined,
                WebkitMask: step.target
                  ? `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100%' height='100%'%3E%3Cdefs%3E%3Cmask id='mask' x='0' y='0' width='100%' height='100%'%3E%3Crect x='0' y='0' width='100%' height='100%' fill='white'/%3E%3Ccircle cx='${tooltipPosition.left}' cy='${tooltipPosition.top - window.scrollY}' r='${step.spotlightRadius || 100}' fill='black'/%3E%3C/mask%3E%3C/defs%3E%3Crect x='0' y='0' width='100%' height='100%' mask='url(%23mask)' fill='black'/%3E%3C/svg%3E")`
                  : undefined,
              }}
            />
          )}

          {/* Tooltip */}
          <motion.div
            variants={tooltipVariants}
            className={cn(
              "absolute z-50 p-5 bg-white dark:bg-gray-900 rounded-lg shadow-xl border border-gray-200 dark:border-gray-800 max-w-sm",
              className
            )}
            style={{
              top: step.target ? tooltipPosition.top : "50%",
              left: step.target ? tooltipPosition.left : "50%",
              transform: step.target ? "translate(-50%, -50%)" : "translate(-50%, -50%)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close button */}
            <button
              onClick={handleSkip}
              className="absolute top-2 right-2 text-gray-400 hover:text-gray-500 dark:text-gray-500 dark:hover:text-gray-400"
              aria-label="Close"
            >
              <X size={16} />
            </button>

            {/* Content */}
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                {step.title}
              </h3>
              <div className="text-gray-600 dark:text-gray-300 text-sm">
                {step.description}
              </div>
            </div>

            {/* Progress indicator */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex space-x-1">
                {steps.map((_, index) => (
                  <div
                    key={index}
                    className={cn(
                      "h-1.5 rounded-full w-6 transition-colors",
                      index === currentStep
                        ? "bg-primary-600 dark:bg-primary-500"
                        : "bg-gray-200 dark:bg-gray-700"
                    )}
                  />
                ))}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400">
                {currentStep + 1} of {steps.length}
              </div>
            </div>

            {/* Actions */}
            <div className="flex justify-between items-center">
              <Button
                variant="ghost"
                onClick={handleSkip}
              >
                {step.actions?.skipLabel || "Skip tour"}
              </Button>
              <div className="flex items-center space-x-2">
                {currentStep > 0 && (
                  <Button
                    variant="outline"
                    onClick={handlePrev}
                    leftIcon={<ChevronLeft size={16} />}
                  >
                    {step.actions?.prevLabel || "Previous"}
                  </Button>
                )}
                <Button
                  variant="primary"
                  onClick={handleNext}
                  rightIcon={currentStep < steps.length - 1 ? <ChevronRight size={16} /> : undefined}
                >
                  {currentStep < steps.length - 1
                    ? step.actions?.nextLabel || "Next"
                    : step.actions?.doneLabel || "Finish"}
                </Button>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}; 