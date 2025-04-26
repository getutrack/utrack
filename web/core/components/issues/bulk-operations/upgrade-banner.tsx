"use client";

// ui
import { getButtonStyling } from "@utrack/ui";
// constants
import { MARKETING_UTRACK_ONE_PAGE_LINK } from "@/constants/common";
// helpers
import { cn } from "@/helpers/common.helper";
import React from "react";

type Props = {
  selectedIssueIds?: string[];
  className?: string;
};

export const BulkOperationsUpgradeBanner: React.FC<Props> = () => {
  // Return null to not show any upgrade banner
  return null;
};
