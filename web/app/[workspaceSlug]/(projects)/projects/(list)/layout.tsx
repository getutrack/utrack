"use client";

import { ReactNode } from "react";
// components
import { AppHeader, ContentWrapper } from "@/components/core";
// local components
import { ProjectsListHeader } from "@/utrack-web/components/projects/header";
import { ProjectsListMobileHeader } from "@/utrack-web/components/projects/mobile-header";
export default function ProjectListLayout({ children }: { children: ReactNode }) {
  return (
    <>
      <AppHeader header={<ProjectsListHeader />} mobileHeader={<ProjectsListMobileHeader />} />
      <ContentWrapper>{children}</ContentWrapper>
    </>
  );
}
