"use client";

import { FC } from "react";
import { CheckCircle } from "lucide-react";
import { Tab } from "@headlessui/react";
// helpers
import { cn } from "@/helpers/common.helper";

export type ProPlanUpgradeProps = {
  basePlan: "Free" | "One";
  features: string[];
  verticalFeatureList?: boolean;
  extraFeatures?: string | React.ReactNode;
};

type TProPiceFrequency = "month" | "year";

type TProPlanPrice = {
  key: string;
  currency: string;
  price: number;
  recurring: TProPiceFrequency;
};

// constants
export const calculateYearlyDiscount = (monthlyPrice: number, yearlyPricePerMonth: number): number => {
  const monthlyCost = monthlyPrice * 12;
  const yearlyCost = yearlyPricePerMonth * 12;
  const amountSaved = monthlyCost - yearlyCost;
  const discountPercentage = (amountSaved / monthlyCost) * 100;
  return Math.floor(discountPercentage);
};

const PRO_PLAN_PRICES: TProPlanPrice[] = [
  { key: "monthly", currency: "$", price: 8, recurring: "month" },
  { key: "yearly", currency: "$", price: 6, recurring: "year" },
];

export const ProPlanUpgrade: FC<ProPlanUpgradeProps> = () => {
  // Return null to hide the upgrade UI
  return null;
};
