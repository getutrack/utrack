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

export const PaidPlanUpgradeModal: FC<PaidPlanUpgradeModalProps> = (props) => {
  const { isOpen, handleClose } = props;

  return (
    <ModalCore isOpen={isOpen} handleClose={handleClose} width={EModalWidth.VIXL} className="rounded-2xl">
      <div className="p-10 max-h-[90vh] overflow-auto">
        <div className="grid grid-cols-12 gap-6">
          <div className="col-span-12 md:col-span-4">
            <div className="text-3xl font-bold leading-8 flex">Upgrade to a paid plan and unlock missing features.</div>
            <div className="mt-4 mb-12">
              <p className="text-sm mb-4 pr-8 text-custom-text-100">
                Active Cycles, time tracking, bulk ops, and other features are waiting for you on one of our paid plans.
                Upgrade today to unlock features your teams need yesterday.
              </p>
            </div>
            {/* Remove Free plan details */}
          </div>
          <div className="col-span-12 md:col-span-4">
            <ProPlanUpgrade
              basePlan="One"
              features={PRO_PLAN_FEATURES}
              verticalFeatureList
              extraFeatures={
                <p className="pt-1.5 text-center text-xs text-custom-primary-200 font-semibold underline">
                  <a href="https://getutrack.io/pro" target="_blank">
                    See full features list
                  </a>
                </p>
              }
            />
          </div>
          <div className="col-span-12 md:col-span-4">
            <OnePlanUpgrade
              features={ONE_PLAN_FEATURES}
              verticalFeatureList
              extraFeatures={
                <p className="pt-1.5 text-center text-xs text-custom-primary-200 font-semibold underline">
                  <a href="https://getutrack.io/one" target="_blank">
                    See full features list
                  </a>
                </p>
              }
            />
          </div>
        </div>
      </div>
    </ModalCore>
  );
};
