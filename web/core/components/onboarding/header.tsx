import { FC } from "react";
import Image from "next/image";
// images
import BlueYoutrackLogoWithoutText from "@/public/youtrack-logos/blue-without-text.png";
// components
import { OnboardingStepIndicator } from "./step-indicator";

export type OnboardingHeaderProps = {
  currentStep: number;
  totalSteps: number;
};

export const OnboardingHeader: FC<OnboardingHeaderProps> = (props) => {
  const { currentStep, totalSteps } = props;

  return (
    <div className="flex w-full items-center justify-between font-semibold ">
      <div className="flex items-center gap-x-2">
        <Image src={BlueYoutrackLogoWithoutText} height={30} width={30} alt="Youtrack Logo" className="mr-3" />
        <OnboardingStepIndicator currentStep={currentStep} totalSteps={totalSteps} />
      </div>
    </div>
  );
};
