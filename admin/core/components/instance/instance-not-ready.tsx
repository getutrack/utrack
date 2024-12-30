"use client";

import { FC } from "react";
import Image from "next/image";
import Link from "next/link";
import { Button } from "@utrack/ui";
// assets
import UtrackTakeOffImage from "@/public/images/utrack-takeoff.png";

export const InstanceNotReady: FC = () => (
  <div className="h-full w-full relative container px-5 mx-auto flex justify-center items-center">
    <div className="w-auto max-w-2xl relative space-y-8 py-10">
      <div className="relative flex flex-col justify-center items-center space-y-4">
        <h1 className="text-3xl font-bold pb-3">Welcome aboard Utrack!</h1>
        <Image src={UtrackTakeOffImage} alt="Utrack Logo" />
        <p className="font-medium text-base text-onboarding-text-400">
          Get started by setting up your instance and workspace
        </p>
      </div>

      <div>
        <Link href={"/setup/?auth_enabled=0"}>
          <Button size="lg" className="w-full">
            Get started
          </Button>
        </Link>
      </div>
    </div>
  </div>
);
