import { FC } from "react";
import { CheckCircle } from "lucide-react";
// helpers
import { cn } from "@/helpers/common.helper";

export type OnePlanUpgradeProps = {
  features: string[];
  verticalFeatureList?: boolean;
  extraFeatures?: string | React.ReactNode;
};

export const OnePlanUpgrade: FC<OnePlanUpgradeProps> = () => {
  // Return null to hide the upgrade UI
  return null;
};
