"use client";

type TPageHeader = {
  title?: string;
  description?: string;
};

export const PageHeader: React.FC<TPageHeader> = (props) => {
  const { title = "God Mode - Youtrack", description = "Youtrack god mode" } = props;

  return (
    <>
      <title>{title}</title>
      <meta name="description" content={description} />
    </>
  );
};
