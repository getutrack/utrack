"use client";

import { Home } from "lucide-react";
// ui
import { Breadcrumbs, Header } from "@utrack/ui";
// components
import { BreadcrumbLink } from "@/components/common";

export const WorkspaceDashboardHeader = () => {
  return (
    <>
      <Header>
        <Header.LeftItem>
          <div>
            <Breadcrumbs>
              <Breadcrumbs.BreadcrumbItem
                type="text"
                link={<BreadcrumbLink label="Home" icon={<Home className="h-4 w-4 text-custom-text-300" />} />}
              />
            </Breadcrumbs>
          </div>
        </Header.LeftItem>
      </Header>
    </>
  );
};
