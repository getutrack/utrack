import { FC } from "react";
// types
import { CircleX } from "lucide-react";
// services
import { EModalWidth, ModalCore } from "@utrack/ui";
// utrack web components
import { cn } from "@/helpers/common.helper";
// local components
import { OnePlanUpgrade } from "./one-plan-upgrade";
import { ProPlanUpgrade } from "./pro-plan-upgrade";

const PRO_PLAN_FEATURES = [
  "More Cycles features",
  "Full Time Tracking + Bulk Ops",
  "Workflow manager",
  "Automations",
  "Popular integrations",
  "Utrack AI",
];

const ONE_PLAN_FEATURES = [
  "OIDC + SAML for SSO",
  "Active Cycles",
  "Real-time collab + public views and page",
  "Link pages in issues and vice-versa",
  "Time-tracking + limited bulk ops",
  "Docker, Kubernetes and more",
];

const FREE_PLAN_UPGRADE_FEATURES = [
  "OIDC + SAML for SSO",
  "Time tracking and bulk ops",
  "Integrations",
  "Public views and pages",
];

export type PaidPlanUpgradeModalProps = {
  isOpen: boolean;
  handleClose: () => void;
};

export const PaidPlanUpgradeModal: FC<PaidPlanUpgradeModalProps> = () => {
  // Return null to hide the upgrade modal
  return null;
};
