"use client";

import React, { useEffect, useRef } from "react";

interface FocusTrapProps {
  children: React.ReactNode;
  active?: boolean;
  returnFocusOnDeactivate?: boolean;
  initialFocusRef?: React.RefObject<HTMLElement>;
}

export const FocusTrap = ({
  children,
  active = true,
  returnFocusOnDeactivate = true,
  initialFocusRef,
}: FocusTrapProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const previouslyFocusedElement = useRef<HTMLElement | null>(null);

  // Store previously focused element
  useEffect(() => {
    if (active && returnFocusOnDeactivate) {
      previouslyFocusedElement.current = document.activeElement as HTMLElement;
    }
  }, [active, returnFocusOnDeactivate]);

  // Set initial focus
  useEffect(() => {
    if (active) {
      const elementToFocus = initialFocusRef?.current || findFirstFocusableElement(containerRef.current);
      if (elementToFocus) {
        requestAnimationFrame(() => {
          elementToFocus.focus();
        });
      }
    }
  }, [active, initialFocusRef]);

  // Return focus on unmount
  useEffect(() => {
    return () => {
      if (returnFocusOnDeactivate && previouslyFocusedElement.current) {
        previouslyFocusedElement.current.focus();
      }
    };
  }, [returnFocusOnDeactivate]);

  // Handle tab and shift+tab
  useEffect(() => {
    if (!active) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Tab" || !containerRef.current) return;

      const focusableElements = getFocusableElements(containerRef.current);
      if (focusableElements.length === 0) return;

      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];

      // Handle shift+tab
      if (event.shiftKey) {
        if (document.activeElement === firstElement) {
          lastElement.focus();
          event.preventDefault();
        }
      }
      // Handle tab
      else {
        if (document.activeElement === lastElement) {
          firstElement.focus();
          event.preventDefault();
        }
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [active]);

  return <div ref={containerRef}>{children}</div>;
};

// Helper functions
const getFocusableElements = (element: HTMLElement): HTMLElement[] => {
  return Array.from(
    element.querySelectorAll(
      'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'
    )
  ) as HTMLElement[];
};

const findFirstFocusableElement = (element: HTMLElement | null): HTMLElement | null => {
  if (!element) return null;
  const focusableElements = getFocusableElements(element);
  return focusableElements.length > 0 ? focusableElements[0] : null;
}; 