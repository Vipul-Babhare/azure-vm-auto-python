provider "azurerm" {
  features {}
  subscription_id = "ab16a6e7-fa55-4581-b4dd-5602b818cd8e"
  use_cli         = true
}

variable "ssh_public_key" {
  type        = string
  description = "SSH Public Key for the VM"
}

variable "allowed_ssh_cidr" {
  description = "CIDR allowed to SSH into VM"
  type        = string
  default     = "0.0.0.0/0"
}

variable "allowed_ip_for_tf_serving" {
  description = "CIDR allowed to access TF Serving"
  type        = string
  default     = "0.0.0.0/0"
}

variable "allowed_ip_for_streamlit" {
  description = "CIDR allowed to access Streamlit app"
  type        = string
  default     = "0.0.0.0/0"
}

variable "allowed_ip_for_grafana" {
  description = "CIDR allowed to access Grafana"
  type        = string
  default     = "0.0.0.0/0"
}

variable "allowed_ip_for_prometheus" {
  description = "CIDR allowed to access Prometheus"
  type        = string
  default     = "0.0.0.0/0"
}

# Reference existing Resource Group
data "azurerm_resource_group" "example" {
  name = "test-vm-groupterraf"
}

# Reference existing Virtual Network
data "azurerm_virtual_network" "example" {
  name                = "test-vnet"
  resource_group_name = data.azurerm_resource_group.example.name
}

# Reference existing Subnet
data "azurerm_subnet" "example" {
  name                 = "test-subnet"
  virtual_network_name = data.azurerm_virtual_network.example.name
  resource_group_name  = data.azurerm_resource_group.example.name
}

# Reference existing NSG
data "azurerm_network_security_group" "example" {
  name                = "test-nsg"
  resource_group_name = data.azurerm_resource_group.example.name
}

# --- Open Streamlit Port ---
resource "azurerm_network_security_rule" "streamlit" {
  name                        = "Allow-Streamlit"
  priority                    = 400
  direction                   = "Inbound"
  access                      = "Allow"
  protocol                    = "Tcp"
  source_port_range           = "*"
  destination_port_ranges     = ["8502"]
  source_address_prefixes     = [var.allowed_ip_for_streamlit]
  destination_address_prefix  = "*"
  resource_group_name         = data.azurerm_resource_group.example.name
  network_security_group_name = data.azurerm_network_security_group.example.name
}

# --- Open Grafana Port ---
resource "azurerm_network_security_rule" "grafana" {
  name                        = "Allow-Grafana"
  priority                    = 401
  direction                   = "Inbound"
  access                      = "Allow"
  protocol                    = "Tcp"
  source_port_range           = "*"
  destination_port_ranges     = ["3000"]
  source_address_prefixes     = [var.allowed_ip_for_grafana]
  destination_address_prefix  = "*"
  resource_group_name         = data.azurerm_resource_group.example.name
  network_security_group_name = data.azurerm_network_security_group.example.name
}

# --- Open Prometheus Port ---
resource "azurerm_network_security_rule" "prometheus" {
  name                        = "Allow-Prometheus"
  priority                    = 402
  direction                   = "Inbound"
  access                      = "Allow"
  protocol                    = "Tcp"
  source_port_range           = "*"
  destination_port_ranges     = ["9090"]
  source_address_prefixes     = [var.allowed_ip_for_prometheus]
  destination_address_prefix  = "*"
  resource_group_name         = data.azurerm_resource_group.example.name
  network_security_group_name = data.azurerm_network_security_group.example.name
}

# Reference existing Public IP
data "azurerm_public_ip" "example" {
  name                = "myPublicIP"
  resource_group_name = data.azurerm_resource_group.example.name
}

# Reference existing NIC
data "azurerm_network_interface" "example" {
  name                = "test-nic"
  resource_group_name = data.azurerm_resource_group.example.name
}

# Create VM
resource "azurerm_linux_virtual_machine" "example" {
  name                = "test-vm"
  resource_group_name = data.azurerm_resource_group.example.name
  location            = data.azurerm_resource_group.example.location
  size                = "Standard_B1s"
  admin_username      = "azureuser"
  network_interface_ids = [
    data.azurerm_network_interface.example.id,
  ]

  admin_ssh_key {
    username   = "azureuser"
    public_key = var.ssh_public_key
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts"
    version   = "latest"
  }
}

output "public_ip" {
  value       = data.azurerm_public_ip.example.ip_address
  description = "Public IP of the Azure VM"
  sensitive   = false
}
